"""
Flux2Klein Local 4-GPU Inference Script

Local inference on 4x A100 GPUs for testing specific checkpoints.
Checkpoints: 28500, 27000, 25000, 23000
CFG values: 4.0, 3.5
Steps: 50

Usage:
    accelerate launch --num_processes=4 test/test_flux2klein_local_4gpu.py

Output saved to: /home/v-yuxluo/data/inference_results/0129_flux9B_sample/
"""

import argparse
import os
import sys
import gc
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Initialize accelerate state
from accelerate import PartialState

state = PartialState()

from mmengine import Config
from OpenSciDraw.utils.model_factory import (
    ModelFactory,
    get_pipeline_class,
)


# Hardcoded settings for local test
CHECKPOINT_BASE = "/home/v-yuxluo/data/experiments/0129_flux9B_sample"
OUTPUT_BASE = "/home/v-yuxluo/data/inference_results/0129_flux9B_sample"
CONFIG_PATH = "/home/v-yuxluo/WORK_local/ArXivQwenImage/configs/260124/flux2klein_fulltune_50000_amlt_ga4.py"
CHECKPOINT_STEPS = [28500, 27000, 25000, 23000]
GUIDANCE_SCALES = [4.0, 3.5]
NUM_INFERENCE_STEPS = 50
SEED = 42
MAX_SEQUENCE_LENGTH = 1024


def get_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    return dtype_map[dtype_str]


def load_pipeline(config, checkpoint_path: str, device, weight_dtype: torch.dtype):
    """Load pipeline from checkpoint."""
    model_type = getattr(config, 'model_type', 'Flux2Klein')
    config.weight_dtype = weight_dtype
    
    model_factory = ModelFactory(config)
    
    tokenizer = model_factory.load_tokenizer()
    text_encoder = model_factory.load_text_encoder()
    text_encoder = text_encoder.to(device=device, dtype=weight_dtype)
    text_encoder.eval()
    
    vae = model_factory.load_vae()
    vae = vae.to(device=device, dtype=weight_dtype)
    vae.eval()
    
    scheduler = model_factory.load_scheduler()
    
    # Load fine-tuned transformer
    TransformerClass = model_factory.TransformerClass
    transformer = TransformerClass.from_pretrained(
        checkpoint_path,
        torch_dtype=weight_dtype,
    )
    transformer = transformer.to(device=device, dtype=weight_dtype)
    transformer.eval()
    
    PipelineClass = get_pipeline_class(model_type)
    pipeline = PipelineClass(
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    
    return pipeline


def save_image(image: Image.Image, output_dir: str, prompt_idx: int, prompt_text: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt_text[:50])
    safe_prompt = safe_prompt.strip().replace(' ', '_')
    filename = f"prompt{prompt_idx:02d}_{safe_prompt}.png"
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)
    return filepath


def run_inference_for_checkpoint(
    pipeline, 
    validation_prompts, 
    resolution_list, 
    output_dir,
    device,
    weight_dtype,
    guidance_scale,
    num_inference_steps,
    max_sequence_length,
    seed,
    rank,
    world_size
):
    """Run inference for a single checkpoint with a specific guidance scale."""
    # Distribute prompts across GPUs
    prompts_per_gpu = []
    for i, (prompt, res) in enumerate(zip(validation_prompts, resolution_list)):
        if i % world_size == rank:
            prompts_per_gpu.append((i, prompt, res))
    
    os.makedirs(output_dir, exist_ok=True)
    
    for prompt_idx, prompt, (width, height) in prompts_per_gpu:
        generator = torch.Generator(device=device).manual_seed(seed + prompt_idx)
        
        try:
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=weight_dtype):
                    output = pipeline(
                        prompt=prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        max_sequence_length=max_sequence_length,
                        output_type="pil",
                    )
                    image = output.images[0]
            
            filepath = save_image(image, output_dir, prompt_idx, prompt)
            print(f"[Rank {rank}] Saved: {filepath}")
            
        except Exception as e:
            print(f"[Rank {rank}] Error on prompt {prompt_idx}: {e}")
            import traceback
            traceback.print_exc()
            # Save placeholder
            placeholder = Image.new('RGB', (width, height), color='gray')
            save_image(placeholder, output_dir, prompt_idx, f"error_{prompt[:20]}")
        
        gc.collect()
        torch.cuda.empty_cache()


def cleanup_pipeline(pipeline):
    """Clean up pipeline to free GPU memory."""
    del pipeline.transformer
    del pipeline.text_encoder
    del pipeline.vae
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


def main():
    rank = state.process_index
    world_size = state.num_processes
    device = state.device
    weight_dtype = torch.bfloat16
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Flux2Klein Local 4-GPU Inference Test")
        print(f"{'='*60}")
        print(f"  World Size: {world_size} GPUs")
        print(f"  Checkpoint Base: {CHECKPOINT_BASE}")
        print(f"  Checkpoints: {CHECKPOINT_STEPS}")
        print(f"  Guidance Scales: {GUIDANCE_SCALES}")
        print(f"  Inference Steps: {NUM_INFERENCE_STEPS}")
        print(f"  Output: {OUTPUT_BASE}")
        print(f"{'='*60}\n")
    
    # Load config
    config = Config.fromfile(CONFIG_PATH)
    validation_prompts = getattr(config, 'validation_prompts', [])
    resolution_list = getattr(config, 'resolution_list', [[1024, 1024]] * len(validation_prompts))
    
    if len(resolution_list) < len(validation_prompts):
        last_res = resolution_list[-1] if resolution_list else [1024, 1024]
        resolution_list = resolution_list + [last_res] * (len(validation_prompts) - len(resolution_list))
    
    if rank == 0:
        print(f"Total prompts per checkpoint: {len(validation_prompts)}")
    
    # Process each checkpoint
    for step in CHECKPOINT_STEPS:
        checkpoint_path = f"{CHECKPOINT_BASE}/checkpoint-{step}/transformer"
        
        # Check if checkpoint exists
        if rank == 0:
            if not os.path.exists(checkpoint_path):
                print(f"\n[SKIP] Checkpoint {step} not found: {checkpoint_path}")
                continue
            print(f"\n{'='*60}")
            print(f"Processing checkpoint-{step}")
            print(f"{'='*60}")
        
        state.wait_for_everyone()
        
        # Skip if not found (all ranks need to agree)
        if not os.path.exists(checkpoint_path):
            continue
        
        try:
            # Serial model loading (one GPU at a time) to avoid OOM
            pipeline = None
            for load_rank in range(world_size):
                if rank == load_rank:
                    print(f"[Rank {rank}] Loading checkpoint-{step}...")
                    pipeline = load_pipeline(config, checkpoint_path, device, weight_dtype)
                    print(f"[Rank {rank}] Loaded!")
                state.wait_for_everyone()
            
            # Run inference for each guidance scale
            for cfg in GUIDANCE_SCALES:
                output_dir = f"{OUTPUT_BASE}/checkpoint-{step}/cfg_{cfg}"
                
                if rank == 0:
                    print(f"\n  Running with CFG={cfg}...")
                
                state.wait_for_everyone()
                
                run_inference_for_checkpoint(
                    pipeline=pipeline,
                    validation_prompts=validation_prompts,
                    resolution_list=resolution_list,
                    output_dir=output_dir,
                    device=device,
                    weight_dtype=weight_dtype,
                    guidance_scale=cfg,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                    seed=SEED,
                    rank=rank,
                    world_size=world_size,
                )
                
                state.wait_for_everyone()
                
                if rank == 0:
                    print(f"  CFG={cfg} completed!")
            
            # Cleanup after each checkpoint
            if rank == 0:
                print(f"\n  Cleaning up checkpoint-{step}...")
            
            cleanup_pipeline(pipeline)
            state.wait_for_everyone()
            
            if rank == 0:
                print(f"  Checkpoint {step} finished!")
            
        except Exception as e:
            print(f"[Rank {rank}] Error processing checkpoint {step}: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            continue
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"All checkpoints processed!")
        print(f"Results saved to: {OUTPUT_BASE}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
