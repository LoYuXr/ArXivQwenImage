"""
Flux2Klein Full Fine-tuning Model Inference Script

This script loads a fully fine-tuned Flux2Klein model and generates images
based on validation_prompts and resolution_list from a config file.

Usage:
    python test/test_flux2klein_fulltune_inference.py \
        --config configs/260124/flux2klein_fulltune_50000_amlt_ga1.py \
        --checkpoint_path /path/to/checkpoint \
        --output_dir ./output_images \
        --device cuda \
        --num_inference_steps 28 \
        --guidance_scale 3.5

Arguments:
    --config: Path to the config file containing validation_prompts and resolution_list
    --checkpoint_path: Path to the fine-tuned transformer checkpoint directory
    --output_dir: Directory to save generated images
    --device: Device to use (cuda/cpu)
    --num_inference_steps: Number of denoising steps (default: 28)
    --guidance_scale: Guidance scale for inference (default: 3.5)
    --seed: Random seed for reproducibility (optional)
    --max_sequence_length: Maximum sequence length for text encoding (default: 1024)
"""

import argparse
import os
import sys
import gc
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Initialize accelerate state before importing ModelFactory (required for accelerate logging)
from accelerate import PartialState
PartialState()

from mmengine import Config
from OpenSciDraw.utils.model_factory import (
    ModelFactory,
    get_pipeline_class,
    get_model_spec,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate images using a fine-tuned Flux2Klein model"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file containing validation_prompts and resolution_list",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        default=None,
        help="Path to the fine-tuned transformer checkpoint directory",
    )
    parser.add_argument(
        "--skip_finetuned_weights",
        action="store_true",
        help="Skip loading fine-tuned weights (use original pretrained model for testing)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference (default: cuda)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="Number of denoising steps (default: 28)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for inference (default: 3.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=1024,
        help="Maximum sequence length for text encoding (default: 1024)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Data type for inference (default: bf16)",
    )
    
    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return dtype_map[dtype_str]


def save_image(image: Image.Image, output_dir: str, prompt_idx: int, prompt_text: str) -> str:
    """Save a single image to disk with a descriptive filename."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a safe filename from the prompt
    safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt_text[:50])
    safe_prompt = safe_prompt.strip().replace(' ', '_')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prompt{prompt_idx:02d}_{safe_prompt}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    image.save(filepath)
    return filepath


def load_config(config_path: str):
    """Load and parse the config file directly using mmengine Config."""
    config = Config.fromfile(config_path)
    return config


def load_pipeline(config, checkpoint_path: str, device: str, weight_dtype: torch.dtype, skip_finetuned_weights: bool = False):
    """Load the Flux2Klein pipeline with fine-tuned weights.
    
    Uses ModelFactory for base components and from_pretrained for fine-tuned transformer.
    Supports loading directly from blob storage paths.
    """
    print(f"\n{'='*60}")
    print("Loading Flux2Klein Pipeline")
    print(f"{'='*60}")
    
    # Get model type from config
    model_type = getattr(config, 'model_type', 'Flux2Klein')
    print(f"  Model Type: {model_type}")
    print(f"  Base Model: {config.pretrained_model_name_or_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    print(f"  Dtype: {weight_dtype}")
    print(f"  Skip fine-tuned weights: {skip_finetuned_weights}")
    
    # Set weight_dtype on config for ModelFactory
    config.weight_dtype = weight_dtype
    
    # Use ModelFactory to load base components
    model_factory = ModelFactory(config)
    
    # Load tokenizer
    print("\n  Loading tokenizer...")
    tokenizer = model_factory.load_tokenizer()
    
    # Load text encoder
    print("  Loading text encoder...")
    text_encoder = model_factory.load_text_encoder()
    text_encoder = text_encoder.to(device=device, dtype=weight_dtype)
    text_encoder.eval()
    
    # Load VAE
    print("  Loading VAE...")
    vae = model_factory.load_vae()
    vae = vae.to(device=device, dtype=weight_dtype)
    vae.eval()
    
    # Load scheduler
    print("  Loading scheduler...")
    scheduler = model_factory.load_scheduler()
    
    # Load transformer - either fine-tuned or original
    if skip_finetuned_weights or checkpoint_path is None:
        print("\n  Loading original pretrained transformer...")
        transformer = model_factory.load_transformer()
    else:
        print(f"\n  Loading fine-tuned transformer from: {checkpoint_path}")
        # Directly use from_pretrained - works with blob paths!
        TransformerClass = model_factory.TransformerClass
        transformer = TransformerClass.from_pretrained(
            checkpoint_path,
            torch_dtype=weight_dtype,
        )
        print(f"    ✅ Loaded fine-tuned transformer")
    
    transformer = transformer.to(device=device, dtype=weight_dtype)
    transformer.eval()
    
    # Create pipeline
    print("\n  Creating pipeline...")
    PipelineClass = get_pipeline_class(model_type)
    
    pipeline = PipelineClass(
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    
    print(f"\n{'='*60}")
    print("Pipeline loaded successfully!")
    print(f"{'='*60}\n")
    
    return pipeline


def generate_images(
    pipeline,
    validation_prompts: list,
    resolution_list: list,
    output_dir: str,
    device: str,
    weight_dtype: torch.dtype,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    max_sequence_length: int = 1024,
    seed: int = None,
):
    """Generate images for all validation prompts."""
    
    print(f"\n{'='*60}")
    print("Starting Image Generation")
    print(f"{'='*60}")
    print(f"  Number of prompts: {len(validation_prompts)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Inference steps: {num_inference_steps}")
    print(f"  Guidance scale: {guidance_scale}")
    print(f"  Seed: {seed if seed is not None else 'Random'}")
    print(f"{'='*60}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for prompt_idx, prompt in enumerate(tqdm(validation_prompts, desc="Generating images")):
        # Get resolution
        if prompt_idx < len(resolution_list):
            width, height = resolution_list[prompt_idx]
        else:
            width, height = 1024, 1024
            print(f"  Warning: No resolution for prompt {prompt_idx}, using default 1024x1024")
        
        print(f"\n  Prompt {prompt_idx + 1}/{len(validation_prompts)}:")
        print(f"    Resolution: {width}x{height}")
        print(f"    Text: {prompt[:100]}...")
        
        # Set generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed + prompt_idx)
        
        # Prepare pipeline kwargs
        pipeline_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "output_type": "pil",
        }
        
        if max_sequence_length:
            pipeline_kwargs["max_sequence_length"] = max_sequence_length
        
        try:
            with torch.no_grad():
                with torch.amp.autocast(device, dtype=weight_dtype):
                    output = pipeline(**pipeline_kwargs)
                    images = output.images
            
            # Save image(s)
            if isinstance(images, list):
                for i, img in enumerate(images):
                    filepath = save_image(img, output_dir, prompt_idx, prompt)
                    saved_paths.append(filepath)
                    print(f"    Saved: {filepath}")
            else:
                filepath = save_image(images, output_dir, prompt_idx, prompt)
                saved_paths.append(filepath)
                print(f"    Saved: {filepath}")
                
        except Exception as e:
            print(f"    Error generating image: {e}")
            import traceback
            traceback.print_exc()
            # Create a placeholder gray image
            placeholder = Image.new('RGB', (width, height), color='gray')
            filepath = save_image(placeholder, output_dir, prompt_idx, f"error_{prompt[:20]}")
            saved_paths.append(filepath)
            print(f"    Saved placeholder: {filepath}")
        
        # Clear cache periodically
        if (prompt_idx + 1) % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return saved_paths


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("Flux2Klein Full Fine-tuning Inference")
    print(f"{'='*60}")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Device: {args.device}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Skip fine-tuned weights: {args.skip_finetuned_weights}")
    print(f"{'='*60}\n")
    
    # Validate paths
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    if not args.skip_finetuned_weights and args.checkpoint_path is not None:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    elif not args.skip_finetuned_weights and args.checkpoint_path is None:
        raise ValueError("Either --checkpoint_path must be provided or --skip_finetuned_weights must be set")
    
    # Load config
    print("Loading config...")
    config = load_config(args.config)
    
    # Get validation prompts and resolutions from config
    validation_prompts = getattr(config, 'validation_prompts', None)
    resolution_list = getattr(config, 'resolution_list', None)
    
    if validation_prompts is None:
        raise ValueError("Config must contain 'validation_prompts'")
    
    if resolution_list is None:
        print("Warning: 'resolution_list' not found in config, using default 1024x1024")
        resolution_list = [[1024, 1024]] * len(validation_prompts)
    
    # Ensure resolution_list matches prompts
    if len(resolution_list) < len(validation_prompts):
        print(f"Warning: resolution_list ({len(resolution_list)}) shorter than prompts ({len(validation_prompts)})")
        # Pad with last resolution or default
        last_res = resolution_list[-1] if resolution_list else [1024, 1024]
        resolution_list = resolution_list + [last_res] * (len(validation_prompts) - len(resolution_list))
    
    print(f"\nFound {len(validation_prompts)} validation prompts")
    
    # Get dtype
    weight_dtype = get_dtype(args.dtype)
    
    # Load pipeline
    pipeline = load_pipeline(
        config, 
        args.checkpoint_path, 
        args.device, 
        weight_dtype,
        skip_finetuned_weights=args.skip_finetuned_weights,
    )
    
    # Generate images
    saved_paths = generate_images(
        pipeline=pipeline,
        validation_prompts=validation_prompts,
        resolution_list=resolution_list,
        output_dir=args.output_dir,
        device=args.device,
        weight_dtype=weight_dtype,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        seed=args.seed,
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("Generation Complete!")
    print(f"{'='*60}")
    print(f"  Total images generated: {len(saved_paths)}")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # List all saved files
    print("Saved images:")
    for path in saved_paths:
        print(f"  - {path}")
    
    # Cleanup
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
