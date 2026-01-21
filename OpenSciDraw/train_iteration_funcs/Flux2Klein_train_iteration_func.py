"""
Flux2Klein Training Iteration Functions for OpenSciDraw

Flux2Klein uses Flux2 architecture with Qwen3 text encoder.
Key differences from QwenImage:
1. Uses img_ids and txt_ids for positional encoding (RoPE)
2. Different VAE latent format
3. guidance_scale instead of true_cfg_scale
4. max_sequence_length can be 512 or 1024
"""

from contextlib import nullcontext
import random
import math

import torch
from diffusers.training_utils import compute_loss_weighting_for_sd3

from OpenSciDraw.utils import (
    get_sigmas,
    compute_density_for_timestep_sampling,
)
from OpenSciDraw.registry import TRAIN_ITERATION_FUNCS


def _prepare_latent_image_ids(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype, already_patchified: bool = False):
    """
    Prepare 4D position IDs for image latent tokens.
    Matches official Flux2Klein pipeline format: [T, H, W, L]
    
    Args:
        batch_size: Batch size
        height: Height of latent
        width: Width of latent
        device: Device to create tensor on
        dtype: Data type for the tensor
        already_patchified: If True, height/width are already after patchify (H//2, W//2).
                           If False, will divide by 2 internally.
    
    Returns:
        img_ids: Tensor of shape (batch_size, H*W, 4) for RoPE
                 Columns are [T=0, H_idx, W_idx, L=0]
    """
    # For Flux2, latent is packed with patch_size=2
    if already_patchified:
        latent_h = height
        latent_w = width
    else:
        latent_h = height // 2
        latent_w = width // 2
    
    # Use cartesian_prod like official implementation: [T, H, W, L]
    t = torch.arange(1, device=device)  # [0] - time dimension
    h = torch.arange(latent_h, device=device)
    w = torch.arange(latent_w, device=device)
    l = torch.arange(1, device=device)  # [0] - layer dimension
    
    # Create position IDs: (H*W, 4) with format [T, H, W, L]
    latent_ids = torch.cartesian_prod(t, h, w, l).to(dtype=dtype)
    
    # Expand to batch: (B, H*W, 4)
    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
    
    return latent_ids


def _prepare_text_ids(seq_len: int, batch_size: int, device: torch.device, dtype: torch.dtype):
    """
    Prepare 4D position IDs for text tokens.
    Matches official Flux2Klein pipeline format: [T, H, W, L]
    
    Args:
        seq_len: Sequence length of text tokens
        batch_size: Batch size
        device: Device to create tensor on
        dtype: Data type
    
    Returns:
        txt_ids: Tensor of shape (batch_size, seq_len, 4) for RoPE
                 Columns are [T=0, H=0, W=0, L=idx]
    """
    # Use cartesian_prod like official implementation: [T, H, W, L]
    t = torch.arange(1, device=device)  # [0]
    h = torch.arange(1, device=device)  # [0]
    w = torch.arange(1, device=device)  # [0]
    l = torch.arange(seq_len, device=device)  # [0, 1, 2, ..., seq_len-1]
    
    # Create position IDs: (seq_len, 4) with format [T, H, W, L]
    txt_ids = torch.cartesian_prod(t, h, w, l).to(dtype=dtype)
    
    # Expand to batch: (B, seq_len, 4)
    txt_ids = txt_ids.unsqueeze(0).expand(batch_size, -1, -1)
    
    return txt_ids


def _pack_latents_flux2(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """
    Pack latents for Flux2 Transformer.
    
    Flux2 expects latents in shape (B, seq_len, C) where:
    - B is batch size
    - seq_len = (H // patch_size) * (W // patch_size)
    - C = original_channels * patch_size * patch_size
    
    Args:
        latents: Tensor of shape (B, C, H, W)
        patch_size: Patch size (default 2 for Flux2)
    
    Returns:
        packed: Tensor of shape (B, seq_len, C * patch_size^2)
    """
    B, C, H, W = latents.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"Height {H} and Width {W} must be divisible by patch_size {patch_size}"
    
    # Reshape to patches
    latents = latents.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    latents = latents.permute(0, 2, 4, 1, 3, 5)  # (B, H', W', C, p, p)
    latents = latents.reshape(B, (H // patch_size) * (W // patch_size), C * patch_size * patch_size)
    
    return latents


def _unpack_latents_flux2(
    latents: torch.Tensor, 
    latent_height: int, 
    latent_width: int, 
    patch_size: int = 2,
) -> torch.Tensor:
    """
    Unpack Flux2 Transformer output back to image latent format.
    
    Args:
        latents: Tensor of shape (B, seq_len, C * patch_size^2)
        latent_height: Height of latent (after VAE encoding, before packing)
        latent_width: Width of latent (after VAE encoding, before packing)
        patch_size: Patch size (default 2 for Flux2)
    
    Returns:
        unpacked: Tensor of shape (B, C, latent_height, latent_width)
    """
    B, seq_len, packed_channels = latents.shape
    
    # Calculate packed dimensions
    packed_h = latent_height // patch_size
    packed_w = latent_width // patch_size
    
    # Infer number of channels
    channels = packed_channels // (patch_size * patch_size)
    
    # Reshape back
    latents = latents.reshape(B, packed_h, packed_w, channels, patch_size, patch_size)
    latents = latents.permute(0, 3, 1, 4, 2, 5)  # (B, C, H', p, W', p)
    latents = latents.reshape(B, channels, latent_height, latent_width)
    
    return latents


@TRAIN_ITERATION_FUNCS.register_module()
def Flux2Klein_train_iteration_func_parquet(
    batch,
    vae,
    transformer,
    text_encoding_pipeline,
    noise_scheduler_copy,
    accelerator,
    config,
):
    """
    Training iteration function for Flux2Klein with pre-computed parquet dataset.
    
    Expected batch format:
        {
            "latents": Tensor of shape (B, C, H, W) - pre-computed VAE latents
            "text_embeds": Tensor of shape (B, seq_len, D) - pre-computed text embeddings
            "text_mask": Tensor of shape (B, seq_len) - attention mask for text (optional)
            "bucket_size": Tuple (height, width) - original image size
        }
    
    Flux2Klein Differences from QwenImage:
    1. Uses img_ids and txt_ids for RoPE positional encoding
    2. VAE latent is 4D (B, C, H, W), not 5D
    3. Different packing/unpacking logic
    4. timestep is divided by 1000 in transformer call
    """
    # ================== 1. Get text embeddings ==================
    text_embeddings = batch["text_embeds"]
    text_mask = batch.get("text_mask", None)  # Optional for Flux2Klein
    
    # ================== 2. Get and normalize latents ==================
    model_input_latent = batch["latents"]
    
    # Handle DDP wrapped model - access dtype via .module if needed
    transformer_dtype = transformer.module.dtype if hasattr(transformer, 'module') else transformer.dtype
    model_input = model_input_latent.to(dtype=transformer_dtype)
    
    # Flux2 latent normalization using VAE BatchNorm parameters
    # The pre-computed latents are NOT normalized, so we need to apply BN normalization here
    # Note: We need to patchify first (like _patchify_latents), then normalize
    bsz, C, H, W = model_input.shape
    
    # Patchify: (B, C, H, W) -> (B, C*4, H//2, W//2)
    model_input = model_input.view(bsz, C, H // 2, 2, W // 2, 2)
    model_input = model_input.permute(0, 1, 3, 5, 2, 4)  # (B, C, 2, 2, H', W')
    model_input = model_input.reshape(bsz, C * 4, H // 2, W // 2)
    
    # Apply VAE BatchNorm normalization
    if vae is not None and hasattr(vae, 'bn'):
        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(model_input.device, model_input.dtype)
        latents_bn_var = vae.bn.running_var.view(1, -1, 1, 1).to(model_input.device, model_input.dtype)
        bn_eps = vae.config.batch_norm_eps if hasattr(vae.config, 'batch_norm_eps') else 1e-4
        latents_bn_std = torch.sqrt(latents_bn_var + bn_eps)
        model_input = (model_input - latents_bn_mean) / latents_bn_std
    
    bsz = model_input.shape[0]
    batch_bucket_h, batch_bucket_w = batch["bucket_size"][0], batch["bucket_size"][1]
    
    # Latent dimensions (after patchify)
    latent_h = model_input.shape[2]
    latent_w = model_input.shape[3]
    latent_channels = model_input.shape[1]
    
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
    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
    
    # ================== 5. Add noise (flow matching) ==================
    with torch.no_grad():
        sigmas = get_sigmas(
            timesteps, 
            device=model_input.device, 
            noise_scheduler_copy=noise_scheduler_copy, 
            n_dim=model_input.ndim, 
            dtype=model_input.dtype
        )
        
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
    
    # ================== 6. Flatten latents for Transformer ==================
    # After patchify, latents are (B, C*4, H', W'), need to flatten to (B, seq_len, C*4)
    # where seq_len = H' * W'
    packed_noisy_input = noisy_model_input.flatten(2).transpose(1, 2)  # (B, H'*W', C*4)
    
    # ================== 7. Prepare position IDs ==================
    with torch.no_grad():
        # Image position IDs: (B, img_seq_len, 4)
        # Note: latent_h, latent_w are already patchified dimensions
        img_ids = _prepare_latent_image_ids(
            batch_size=bsz,
            height=latent_h,
            width=latent_w,
            device=model_input.device,
            dtype=transformer_dtype,
            already_patchified=True,
        )
        
        # Text position IDs: Use pre-computed if available, else generate
        if "text_ids" in batch and batch["text_ids"] is not None:
            # Use pre-computed text_ids from parquet (from encode_prompt)
            txt_ids = batch["text_ids"].to(device=model_input.device, dtype=transformer_dtype)
        else:
            # Fallback: generate text position IDs
            txt_seq_len = text_embeddings.shape[1]
            txt_ids = _prepare_text_ids(
                seq_len=txt_seq_len,
                batch_size=bsz,
                device=model_input.device,
                dtype=transformer_dtype,
            )
    
    # ================== 8. Prepare guidance ==================
    # For CFG-distilled models, guidance is usually not needed during training
    # For non-distilled models, you may need to add guidance embedding
    guidance = None
    if hasattr(config, 'guidance_scale') and config.guidance_scale > 1.0:
        guidance = torch.full(
            (bsz,), 
            config.guidance_scale, 
            device=model_input.device, 
            dtype=transformer_dtype
        )
    
    # ================== 9. Transformer forward ==================
    # NOTE: Flux2 transformer divides timestep by 1000 internally
    model_pred = transformer(
        hidden_states=packed_noisy_input,
        encoder_hidden_states=text_embeddings,
        timestep=timesteps / 1000,  # Flux2 expects timestep in [0, 1]
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        return_dict=False,
    )[0]
    
    # ================== 10. Unpack prediction ==================
    # model_pred is (B, seq_len, C*4), reshape back to (B, C*4, H', W')
    # where seq_len = H' * W', same shape as model_input after patchify
    model_pred = model_pred.transpose(1, 2).reshape(bsz, latent_channels, latent_h, latent_w)
    
    # ================== 11. Compute loss ==================
    # Flow matching target: v = noise - latent
    target = noise - model_input
    
    # Loss weighting
    weighting = compute_loss_weighting_for_sd3(weighting_scheme=config.weighting_scheme, sigmas=sigmas)
    
    # MSE loss
    loss = (model_pred.float() - target.float()) ** 2
    
    # Reduce over spatial dimensions, weighted
    loss = torch.mean((weighting.float() * loss).reshape(bsz, -1), dim=1)
    
    bs = loss.shape[0]
    loss = loss.mean()
    
    return loss, bs


@TRAIN_ITERATION_FUNCS.register_module()
def Flux2Klein_train_iteration_func(
    batch,
    vae,
    transformer,
    text_encoding_pipeline,
    noise_scheduler_copy,
    accelerator,
    config,
):
    """
    Training iteration function for Flux2Klein with online VAE/text encoding.
    
    Expected batch format:
        {
            "pixel_values": Tensor of shape (B, 3, H, W)
            "caption": List of strings
        }
    """
    # Handle DDP wrapped model - access dtype via .module if needed
    transformer_dtype = transformer.module.dtype if hasattr(transformer, 'module') else transformer.dtype
    
    # ================== 1. Text Encoding (Online) ==================
    prompts = batch["caption"]
    
    with torch.no_grad():
        # For Flux2Klein, use the text_encoding_pipeline to get embeddings
        prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
            prompt=prompts,
            max_sequence_length=config.max_sequence_length,
            device=accelerator.device,
        )
    
    prompt_embeds = prompt_embeds.to(dtype=transformer_dtype, device=accelerator.device)
    text_ids = text_ids.to(device=accelerator.device)
    
    # ================== 2. Image Encoding (VAE Online) ==================
    pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=vae.dtype)
    
    bsz = pixel_values.shape[0]
    current_h, current_w = pixel_values.shape[2], pixel_values.shape[3]
    
    with torch.no_grad():
        # VAE Encode
        latents = vae.encode(pixel_values).latent_dist.sample()
    
    # Latent dimensions
    latent_h = latents.shape[2]
    latent_w = latents.shape[3]
    latent_channels = latents.shape[1]
    
    # ================== 3. Normalization ==================
    # Flux2 may or may not need normalization depending on VAE
    # Add if needed: model_input = (latents - mean) * std
    model_input = latents.to(dtype=transformer_dtype)
    
    # ================== 4. Generate noise ==================
    noise = torch.randn_like(model_input)
    
    # ================== 5. Sample timesteps ==================
    u = compute_density_for_timestep_sampling(
        weighting_scheme=config.weighting_scheme,
        batch_size=bsz,
        logit_mean=config.logit_mean,
        logit_std=config.logit_std,
        mode_scale=config.mode_scale,
    )
    
    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
    
    # ================== 6. Add noise (flow matching) ==================
    with torch.no_grad():
        sigmas = get_sigmas(
            timesteps, 
            device=model_input.device, 
            noise_scheduler_copy=noise_scheduler_copy, 
            n_dim=model_input.ndim, 
            dtype=model_input.dtype
        )
        
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
    
    # ================== 7. Pack latents for Transformer ==================
    packed_noisy_input = _pack_latents_flux2(noisy_model_input, patch_size=2)
    
    # ================== 8. Prepare image position IDs ==================
    with torch.no_grad():
        img_ids = _prepare_latent_image_ids(
            batch_size=bsz,
            height=latent_h,
            width=latent_w,
            device=model_input.device,
            dtype=transformer_dtype,
        )
    
    # ================== 9. Prepare guidance ==================
    guidance = None
    if hasattr(config, 'guidance_scale') and config.guidance_scale > 1.0:
        guidance = torch.full(
            (bsz,), 
            config.guidance_scale, 
            device=model_input.device, 
            dtype=transformer_dtype
        )
    
    # ================== 10. Transformer forward ==================
    model_pred = transformer(
        hidden_states=packed_noisy_input,
        encoder_hidden_states=prompt_embeds,
        timestep=timesteps / 1000,
        img_ids=img_ids,
        txt_ids=text_ids,
        guidance=guidance,
        return_dict=False,
    )[0]
    
    # ================== 11. Unpack prediction ==================
    model_pred = _unpack_latents_flux2(
        model_pred,
        latent_height=latent_h,
        latent_width=latent_w,
        patch_size=2,
    )
    
    # ================== 12. Compute loss ==================
    target = noise - model_input
    
    weighting = compute_loss_weighting_for_sd3(weighting_scheme=config.weighting_scheme, sigmas=sigmas)
    
    loss = (model_pred.float() - target.float()) ** 2
    loss = torch.mean((weighting.float() * loss).reshape(bsz, -1), dim=1)
    
    bs = loss.shape[0]
    loss = loss.mean()
    
    return loss, bs
