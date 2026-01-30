"""
QwenImage Full Fine-tuning Training Iteration Functions for OpenSciDraw

Full fine-tuning version - directly train transformer parameters without LoRA.
VAE and text encoder are offloaded and frozen.

QwenImage Key Features:
1. Uses 5D latents: (B, C, T, H, W) where T is temporal dimension (usually 1)
2. Uses img_shapes for multi-aspect-ratio training
3. pack_latents/unpack_latents for handling the latent format
4. ~20B parameters - requires DeepSpeed ZeRO-3 or FSDP for memory efficiency

Reference: Qwen_Image_train_iteration_func.py
"""

from contextlib import nullcontext
import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_loss_weighting_for_sd3

from OpenSciDraw.utils import (
    get_sigmas,
    compute_density_for_timestep_sampling,
    pack_latents,
    unpack_latents,
)
from OpenSciDraw.registry import TRAIN_ITERATION_FUNCS


@TRAIN_ITERATION_FUNCS.register_module()
def QwenImage_fulltune_train_iteration(
    batch,
    vae,
    noise_scheduler_copy,
    transformer,
    config,
    accelerator,
    global_step,
    weight_dtype,
):
    """
    Full fine-tuning training iteration for QwenImage.
    
    This function directly trains the transformer without LoRA adapters.
    VAE and text encoder are assumed to be offloaded and frozen.
    
    Args:
        batch: Batch from parquet dataset containing:
            - latents: Pre-computed VAE latents (B, C, T, H, W) - 5D format
            - text_embeds: Pre-computed text embeddings (B, seq_len, dim)
            - text_mask: Attention mask for text (B, seq_len)
            - bucket_size: Tuple (height, width) - original image size
        vae: VAE model (used only for normalization stats)
        noise_scheduler_copy: Noise scheduler for computing sigmas
        transformer: QwenImage transformer model (trainable, ~20B params)
        config: Training configuration
        accelerator: Accelerator for distributed training
        global_step: Current training step
        weight_dtype: Data type for weights (BF16/FP16/FP32)
    
    Returns:
        loss: Computed loss value
    
    QwenImage Differences from Flux2Klein:
    1. 5D latent format: (B, C, T, H, W) vs 4D for Flux2
    2. Uses img_shapes for multi-aspect-ratio instead of img_ids
    3. Uses encoder_hidden_states_mask instead of separate txt_ids
    4. Different normalization: (latent - mean) * std
    """
    # ================== 1. Get text embeddings ==================
    text_embeddings = batch["text_embeds"].to(accelerator.device, dtype=weight_dtype)
    text_embeddings_mask = batch["text_mask"].to(accelerator.device)
    
    # ================== 2. Get and normalize latents ==================
    # QwenImage latents are 5D: (B, C, T, H, W) where T is temporal dimension
    model_input_latent = batch["latents"].to(accelerator.device, dtype=weight_dtype)
    
    # Normalization: (latent - mean) * std
    # Note: For QwenImage, latents_mean and latents_std are from config
    model_input = (model_input_latent - config.latents_mean) * config.latents_std
    model_input = model_input.to(dtype=weight_dtype)
    
    # Get dimensions
    bsz = model_input.shape[0]
    latent_channels = model_input.shape[1]
    # For 5D: (B, C, T, H, W)
    latent_h = model_input.shape[3]
    latent_w = model_input.shape[4]
    
    # Get bucket size for img_shapes
    batch_bucket_h, batch_bucket_w = batch["bucket_size"][0], batch["bucket_size"][1]
    
    # ================== 3. Generate noise ==================
    noise = torch.randn_like(model_input)
    
    # ================== 4. Sample timesteps ==================
    u = compute_density_for_timestep_sampling(
        weighting_scheme=config.weighting_scheme,
        batch_size=bsz,
        logit_mean=config.logit_mean,
        logit_std=config.logit_std,
        mode_scale=config.mode_scale,
    )
    
    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
    timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)
    
    # ================== 5. Add noise (flow matching) ==================
    with torch.no_grad():
        sigmas = get_sigmas(
            timesteps, 
            device=accelerator.device, 
            noise_scheduler_copy=noise_scheduler_copy, 
            n_dim=model_input.ndim, 
            dtype=model_input.dtype
        )
        
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
    
    # ================== 6. Prepare img_shapes for multi-aspect-ratio ==================
    # img_shapes format: list of (T, H_packed, W_packed) tuples
    # For QwenImage, the transformer expects shapes after packing (divided by 2)
    img_shapes = [
        (1, batch_bucket_h // config.vae_scale_factor // 2, batch_bucket_w // config.vae_scale_factor // 2)
    ] * bsz
    
    # ================== 7. Pack latents for transformer ==================
    # QwenImage expects (B, T, C, H, W) for pack_latents
    # Input is (B, C, T, H, W), so we permute first
    noisy_model_input = noisy_model_input.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
    
    packed_noisy_model_input = pack_latents(
        noisy_model_input,
        batch_size=bsz,
        num_channels_latents=latent_channels,
        height=latent_h,
        width=latent_w,
    )
    
    # ================== 8. Transformer forward ==================
    # Get transformer dtype (handle DDP/DeepSpeed wrapped model)
    if hasattr(transformer, 'module'):
        transformer_dtype = transformer.module.dtype
    else:
        transformer_dtype = transformer.dtype if hasattr(transformer, 'dtype') else weight_dtype
    
    # Forward pass
    model_pred = transformer(
        hidden_states=packed_noisy_model_input.to(transformer_dtype),
        encoder_hidden_states=text_embeddings.to(transformer_dtype),
        encoder_hidden_states_mask=text_embeddings_mask,
        timestep=timesteps / 1000,  # Normalize to [0, 1]
        img_shapes=img_shapes,
        return_dict=False,
    )[0]
    
    # ================== 9. Unpack prediction ==================
    # model_pred is packed, need to unpack to match target dimensions
    try:
        model_pred = unpack_latents(
            model_pred, 
            height=latent_h, 
            width=latent_w,
            vae_scale_factor=1  # Already in latent space
        )
    except RuntimeError as e:
        print(f"❌ Unpack Failed!")
        print(f"Expected: {bsz} * {latent_channels} * {latent_h} * {latent_w}")
        print(f"Actual elements in model_pred: {model_pred.numel()}")
        raise e
    
    # ================== 10. Compute loss ==================
    # Flow matching target: v = noise - latent
    # Need to reshape target to match model_pred
    target = noise - model_input  # Still in (B, C, T, H, W) format
    
    # Loss weighting
    weighting = compute_loss_weighting_for_sd3(weighting_scheme=config.weighting_scheme, sigmas=sigmas)
    
    # MSE loss
    loss = (model_pred.float() - target.float()) ** 2
    
    # Reduce over spatial dimensions, weighted
    loss = torch.mean((weighting.float() * loss).reshape(bsz, -1), dim=1)
    
    loss = loss.mean()
    
    # Debug logging (controlled by config)
    verbose_logging = getattr(config, 'verbose_logging', True)
    if verbose_logging and global_step % 100 == 0 and accelerator.is_main_process:
        print(f"\n[Step {global_step}] QwenImage Full Fine-tune Debug:")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Latent shape: {model_input_latent.shape}")
        print(f"  Bucket size: {batch_bucket_h}x{batch_bucket_w}")
        print(f"  Model input mean: {model_input.mean().item():.4f}, std: {model_input.std().item():.4f}")
        print(f"  Noise mean: {noise.mean().item():.4f}, std: {noise.std().item():.4f}")
        print(f"  Model pred mean: {model_pred.mean().item():.4f}, std: {model_pred.std().item():.4f}")
        # Print sigmas info
        sigmas_flat = sigmas.flatten()
        ts_flat = timesteps.flatten() if timesteps.dim() > 0 else timesteps.unsqueeze(0)
        print(f"  Sigmas: {sigmas_flat[:min(4, len(sigmas_flat))].tolist()}")
    
    return loss


@TRAIN_ITERATION_FUNCS.register_module()
def QwenImage_fulltune_validation_iteration(
    batch,
    vae,
    noise_scheduler,
    transformer,
    config,
    accelerator,
    global_step,
    weight_dtype,
):
    """
    Validation iteration for QwenImage full fine-tuning.
    
    Generates images from validation prompts to monitor training progress.
    """
    # TODO: Implement full validation pipeline with QwenImagePipeline
    raise NotImplementedError("Validation iteration for QwenImage full fine-tuning not yet implemented")
