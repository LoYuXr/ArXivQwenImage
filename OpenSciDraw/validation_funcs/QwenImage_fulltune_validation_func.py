"""
QwenImage Full Fine-tuning Validation Function

For DeepSpeed ZeRO-3: Uses GatheredParameters to collect full weights for inference.
For parquet mode: Temporarily loads text_encoder for validation.
"""

import gc
import os
import torch
import wandb
from PIL import Image

from accelerate.logging import get_logger
from diffusers.utils.torch_utils import is_compiled_module

from OpenSciDraw.registry import VALIDATION_FUNCS
from OpenSciDraw.utils.model_factory import get_pipeline_class, get_text_encoder_class, get_tokenizer_class

logger = get_logger(__name__)


def save_images_to_disk(images, save_dir, global_step, prompt_idx, prompt_text):
    """Save validation images to local disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt_text[:50])
    safe_prompt = safe_prompt.strip().replace(' ', '_')
    
    saved_paths = []
    
    if isinstance(images, list):
        for i, img in enumerate(images):
            filename = f"step{global_step:06d}_prompt{prompt_idx:02d}_{i}_{safe_prompt}.png"
            filepath = os.path.join(save_dir, filename)
            img.save(filepath)
            saved_paths.append(filepath)
    else:
        filename = f"step{global_step:06d}_prompt{prompt_idx:02d}_{safe_prompt}.png"
        filepath = os.path.join(save_dir, filename)
        images.save(filepath)
        saved_paths.append(filepath)
    
    return saved_paths


@VALIDATION_FUNCS.register_module()
def QwenImage_fulltune_validation_func(
    vae,
    transformer,
    text_encoder,
    tokenizer,
    scheduler,
    accelerator,
    args,
    global_step=0,
):
    """Validation function for QwenImage full fine-tuning.
    
    Handles QwenImage-specific pipeline and deals with None text_encoder
    for parquet dataset mode.
    """
    if not accelerator.is_main_process:
        return []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running QwenImage validation at step {global_step}...")
    logger.info(f"{'='*60}")
    
    # Check if text_encoder is available
    if text_encoder is None:
        logger.warning("  text_encoder is None (parquet mode). Skipping validation.")
        logger.info("  To enable validation, use a mode that loads text_encoder.")
        return []
    
    PipelineClass = get_pipeline_class('QwenImage')
    
    # Unwrap transformer
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    if is_compiled_module(unwrapped_transformer):
        unwrapped_transformer = unwrapped_transformer._orig_mod
    
    unwrapped_transformer.eval()
    
    # Prepare output directory
    validation_output_dir = os.path.join(
        args.model_output_dir, 
        "validation_images",
        f"step_{global_step:06d}"
    )
    os.makedirs(validation_output_dir, exist_ok=True)
    
    image_logs = []
    all_saved_paths = []
    
    validation_prompts = getattr(args, 'validation_prompts', None)
    if validation_prompts is None:
        validation_prompts = [
            "A scientific diagram showing the structure of a neural network",
            "A chart comparing different machine learning algorithms",
        ]
        args.resolution_list = [(768, 768), (768, 768)]
    
    # Determine dtype
    if args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    
    # Save original devices
    original_vae_device = next(vae.parameters()).device
    original_te_device = next(text_encoder.parameters()).device
    
    try:
        # Move VAE and text_encoder to GPU for inference
        logger.info("  Moving VAE and text_encoder to GPU...")
        vae.to(accelerator.device)
        text_encoder.to(accelerator.device)
        
        # Create pipeline with all components
        pipeline = PipelineClass(
            vae=vae,
            transformer=unwrapped_transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )
        
        pipeline = pipeline.to(accelerator.device)
        pipeline = pipeline.to(dtype=weight_dtype)
        
        # Generate images
        for prompt_idx, prompt in enumerate(validation_prompts):
            logger.info(f"  Generating image {prompt_idx + 1}/{len(validation_prompts)}: {prompt[:80]}...")
            
            if hasattr(args, 'resolution_list') and prompt_idx < len(args.resolution_list):
                width, height = args.resolution_list[prompt_idx]
            else:
                width, height = 768, 768
            
            generator = None
            if hasattr(args, 'seed') and args.seed is not None:
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + prompt_idx)
            
            pipeline_kwargs = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": getattr(args, 'num_inference_steps', 50),
                "guidance_scale": getattr(args, 'validation_guidance_scale', 1.0),
                "generator": generator,
                "output_type": "pil",
            }
            
            try:
                with torch.no_grad():
                    with torch.amp.autocast(accelerator.device.type, dtype=weight_dtype):
                        output = pipeline(**pipeline_kwargs)
                        images = output.images
            except Exception as e:
                logger.error(f"  Error generating image: {e}")
                import traceback
                logger.error(traceback.format_exc())
                images = [Image.new('RGB', (width, height), color='gray')]
            
            saved_paths = save_images_to_disk(
                images=images,
                save_dir=validation_output_dir,
                global_step=global_step,
                prompt_idx=prompt_idx,
                prompt_text=prompt,
            )
            all_saved_paths.extend(saved_paths)
            
            image_logs.append({
                "images": images,
                "prompt": prompt,
                "paths": saved_paths,
            })
            
            logger.info(f"    Saved to: {saved_paths[0]}")
        
        # Cleanup pipeline
        del pipeline
        
    finally:
        # Move VAE and text_encoder back to original devices
        logger.info("  Moving VAE and text_encoder back to CPU...")
        vae.to(original_vae_device)
        text_encoder.to(original_te_device)
    
    # Log to WandB - use fixed keys so images show progress over steps
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            dict_to_log = {}
            for sample_idx, log in enumerate(image_logs):
                images = log["images"]
                prompt = log["prompt"]
                
                # Use fixed key names (validation/sample_X) so WandB shows timeline progression
                if isinstance(images, list):
                    for i, img in enumerate(images):
                        dict_to_log[f"validation/sample_{sample_idx}_{i}"] = wandb.Image(
                            img, caption=f"[Step {global_step}] {prompt[:120]}..."
                        )
                else:
                    dict_to_log[f"validation/sample_{sample_idx}"] = wandb.Image(
                        images, caption=f"[Step {global_step}] {prompt[:120]}..."
                    )
            
            tracker.log(dict_to_log, step=global_step)
    
    logger.info(f"\n  ✅ Validation complete! Saved {len(all_saved_paths)} images to:")
    logger.info(f"     {validation_output_dir}")
    logger.info(f"{'='*60}\n")
    
    unwrapped_transformer.train()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return image_logs


@VALIDATION_FUNCS.register_module()
def QwenImage_fulltune_validation_func_parquet(
    vae,
    transformer,
    text_encoder,
    tokenizer,
    scheduler,
    accelerator,
    args,
    global_step=0,
):
    """Validation function for QwenImage full fine-tuning with parquet dataset.
    
    For ZeRO-3 with parquet mode:
    1. If text_encoder is None, temporarily load it for validation
    2. Use DeepSpeed's GatheredParameters to collect full transformer weights
    3. Run inference and save images
    4. Clean up to free memory
    """
    if not accelerator.is_main_process:
        return []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running QwenImage validation at step {global_step} (parquet mode)...")
    logger.info(f"{'='*60}")
    
    # Import DeepSpeed utilities for ZeRO-3 weight gathering
    try:
        import deepspeed
        from deepspeed.runtime.zero.partition_parameters import GatheredParameters
        has_deepspeed = True
        is_zero3 = hasattr(accelerator.state, 'deepspeed_plugin') and \
                   accelerator.state.deepspeed_plugin is not None and \
                   accelerator.state.deepspeed_plugin.zero_stage == 3
    except ImportError:
        has_deepspeed = False
        is_zero3 = False
    
    # Determine dtype
    if hasattr(args, 'mixed_precision'):
        if args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        elif args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32
    else:
        weight_dtype = torch.bfloat16
    
    # Temporarily load text_encoder if not available
    temp_text_encoder = None
    temp_tokenizer = None
    
    if text_encoder is None:
        logger.info("  Loading text_encoder temporarily for validation...")
        try:
            TextEncoderClass = get_text_encoder_class('QwenImage')
            TokenizerClass = get_tokenizer_class('QwenImage')
            
            pretrained_path = getattr(args, 'pretrained_model_name_or_path', 'Qwen/Qwen-Image-2512')
            cache_dir = getattr(args, 'cache_dir', None)
            
            temp_tokenizer = TokenizerClass.from_pretrained(
                pretrained_path,
                subfolder="tokenizer",
                cache_dir=cache_dir,
            )
            
            temp_text_encoder = TextEncoderClass.from_pretrained(
                pretrained_path,
                subfolder="text_encoder", 
                torch_dtype=weight_dtype,
                cache_dir=cache_dir,
            )
            temp_text_encoder.to(accelerator.device)
            temp_text_encoder.eval()
            
            text_encoder = temp_text_encoder
            tokenizer = temp_tokenizer
            logger.info("  ✅ Text encoder loaded successfully")
        except Exception as e:
            logger.error(f"  Failed to load text_encoder: {e}")
            logger.warning("  Skipping validation.")
            return []
    
    # Get pipeline class
    PipelineClass = get_pipeline_class('QwenImage')
    
    # Prepare output directory
    validation_output_dir = os.path.join(
        args.model_output_dir, 
        "validation_images",
        f"step_{global_step:06d}"
    )
    os.makedirs(validation_output_dir, exist_ok=True)
    
    image_logs = []
    all_saved_paths = []
    
    validation_prompts = getattr(args, 'validation_prompts', None)
    if validation_prompts is None:
        validation_prompts = [
            "A scientific diagram showing the structure of a neural network",
            "A chart comparing different machine learning algorithms",
        ]
        args.resolution_list = [(768, 768), (768, 768)]
    
    try:
        # Move VAE to GPU for inference
        original_vae_device = next(vae.parameters()).device
        vae.to(accelerator.device)
        
        # Get unwrapped transformer
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        if is_compiled_module(unwrapped_transformer):
            unwrapped_transformer = unwrapped_transformer._orig_mod
        
        # For ZeRO-3, we need to gather all parameters before inference
        if is_zero3:
            logger.info("  ZeRO-3 detected: gathering transformer weights for inference...")
            # Get all parameters that need gathering
            params_to_gather = [p for p in unwrapped_transformer.parameters()]
            
            with GatheredParameters(params_to_gather, modifier_rank=0):
                if accelerator.is_main_process:
                    unwrapped_transformer.eval()
                    
                    # Create pipeline
                    pipeline = PipelineClass(
                        vae=vae,
                        transformer=unwrapped_transformer,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        scheduler=scheduler,
                    )
                    pipeline = pipeline.to(accelerator.device)
                    
                    # Generate images
                    for prompt_idx, prompt in enumerate(validation_prompts):
                        logger.info(f"  Generating image {prompt_idx + 1}/{len(validation_prompts)}: {prompt[:60]}...")
                        
                        if hasattr(args, 'resolution_list') and prompt_idx < len(args.resolution_list):
                            width, height = args.resolution_list[prompt_idx]
                        else:
                            width, height = 768, 768
                        
                        generator = None
                        if hasattr(args, 'seed') and args.seed is not None:
                            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + prompt_idx)
                        
                        try:
                            with torch.no_grad():
                                output = pipeline(
                                    prompt=prompt,
                                    width=width,
                                    height=height,
                                    num_inference_steps=getattr(args, 'num_inference_steps', 28),
                                    guidance_scale=getattr(args, 'validation_guidance_scale', 1.0),
                                    generator=generator,
                                    output_type="pil",
                                )
                                images = output.images
                        except Exception as e:
                            logger.error(f"  Error generating image: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            images = [Image.new('RGB', (width, height), color='gray')]
                        
                        # Save images
                        saved_paths = save_images_to_disk(
                            images=images,
                            save_dir=validation_output_dir,
                            global_step=global_step,
                            prompt_idx=prompt_idx,
                            prompt_text=prompt,
                        )
                        all_saved_paths.extend(saved_paths)
                        
                        image_logs.append({
                            "images": images,
                            "prompt": prompt,
                            "paths": saved_paths,
                        })
                        
                        logger.info(f"    Saved to: {saved_paths[0]}")
                    
                    del pipeline
        else:
            # Non-ZeRO-3 mode (simpler)
            unwrapped_transformer.eval()
            
            pipeline = PipelineClass(
                vae=vae,
                transformer=unwrapped_transformer,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler,
            )
            pipeline = pipeline.to(accelerator.device)
            
            for prompt_idx, prompt in enumerate(validation_prompts):
                logger.info(f"  Generating image {prompt_idx + 1}/{len(validation_prompts)}: {prompt[:60]}...")
                
                if hasattr(args, 'resolution_list') and prompt_idx < len(args.resolution_list):
                    width, height = args.resolution_list[prompt_idx]
                else:
                    width, height = 768, 768
                
                generator = None
                if hasattr(args, 'seed') and args.seed is not None:
                    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + prompt_idx)
                
                try:
                    with torch.no_grad():
                        output = pipeline(
                            prompt=prompt,
                            width=width,
                            height=height,
                            num_inference_steps=getattr(args, 'num_inference_steps', 28),
                            guidance_scale=getattr(args, 'validation_guidance_scale', 1.0),
                            generator=generator,
                            output_type="pil",
                        )
                        images = output.images
                except Exception as e:
                    logger.error(f"  Error generating image: {e}")
                    images = [Image.new('RGB', (width, height), color='gray')]
                
                saved_paths = save_images_to_disk(
                    images=images,
                    save_dir=validation_output_dir,
                    global_step=global_step,
                    prompt_idx=prompt_idx,
                    prompt_text=prompt,
                )
                all_saved_paths.extend(saved_paths)
                
                image_logs.append({
                    "images": images,
                    "prompt": prompt,
                    "paths": saved_paths,
                })
                
                logger.info(f"    Saved to: {saved_paths[0]}")
            
            del pipeline
        
        # Move VAE back
        vae.to(original_vae_device)
        
        # Log to WandB
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                dict_to_log = {}
                for sample_idx, log in enumerate(image_logs):
                    images = log["images"]
                    prompt = log["prompt"]
                    
                    if isinstance(images, list):
                        for i, img in enumerate(images):
                            dict_to_log[f"validation/sample_{sample_idx}_{i}"] = wandb.Image(
                                img, caption=f"[Step {global_step}] {prompt[:120]}..."
                            )
                    else:
                        dict_to_log[f"validation/sample_{sample_idx}"] = wandb.Image(
                            images, caption=f"[Step {global_step}] {prompt[:120]}..."
                        )
                
                tracker.log(dict_to_log, step=global_step)
        
        logger.info(f"\n  ✅ Validation complete! Saved {len(all_saved_paths)} images to:")
        logger.info(f"     {validation_output_dir}")
        logger.info(f"{'='*60}\n")
        
    finally:
        # Clean up temporary text encoder
        if temp_text_encoder is not None:
            del temp_text_encoder
            del temp_tokenizer
        
        # Set transformer back to train mode
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        if is_compiled_module(unwrapped_transformer):
            unwrapped_transformer = unwrapped_transformer._orig_mod
        unwrapped_transformer.train()
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return image_logs

