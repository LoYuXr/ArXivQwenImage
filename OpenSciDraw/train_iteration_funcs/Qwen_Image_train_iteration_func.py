from contextlib import nullcontext
import random

import torch
import math
from itertools import combinations

from diffusers.training_utils import (
    compute_loss_weighting_for_sd3,
)

from OpenSciDraw.utils import (
    compute_text_embeddings,
    get_sigmas,
    compute_density_for_timestep_sampling,
    pack_latents,
    unpack_latents,
)
from OpenSciDraw.registry import TRAIN_ITERATION_FUNCS

@TRAIN_ITERATION_FUNCS.register_module()
def Qwen_Image_train_iteration_func_parquet(
    batch,
    vae,
    transformer,
    text_encoding_pipeline,
    noise_scheduler_copy,
    accelerator,
    config,
):  
    '''
     return {
            "latents": latents,
            "text_embeds": padded_embeds,
            "text_mask": padded_masks,
            "captions": [x['caption'] for x in batch],
            "bucket_size": batch[0]['bucket_size'],  # 全部一样
        }
            
    '''
    text_embeddings, text_embeddings_mask =batch["text_embeds"], batch["text_mask"]
    model_input_latent = batch["latents"]
    model_input = (model_input_latent - config.latents_mean) * config.latents_std #(bs, latent_dim, vid_frame, original_h//8, original_w//8)
    model_input = model_input.to(dtype=vae.dtype)
    
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]
    batch_bucket_h, batch_bucket_w = batch["bucket_size"][0], batch["bucket_size"][1]
    
    u = compute_density_for_timestep_sampling(
        weighting_scheme=config.weighting_scheme,
        batch_size=bsz,
        logit_mean=config.logit_mean,
        logit_std=config.logit_std,
        mode_scale=config.mode_scale,
    )

    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()

    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

    with torch.no_grad():
        sigmas = get_sigmas(timesteps, device=model_input.device, noise_scheduler_copy=noise_scheduler_copy, n_dim=model_input.ndim, dtype=model_input.dtype)

        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        # multi-aspect-ratio
        img_shapes = [
            (1, batch_bucket_h // config.vae_scale_factor // 2, batch_bucket_w // config.vae_scale_factor // 2)
        ] * bsz
                
    # pack for latent
    noisy_model_input = noisy_model_input.permute(0, 2, 1, 3, 4)
    packed_noisy_model_input = pack_latents(
        noisy_model_input,
        batch_size=model_input.shape[0],
        num_channels_latents=model_input.shape[1],
        height=model_input.shape[3],
        width=model_input.shape[4],
    )

    model_pred = transformer(
        hidden_states=packed_noisy_model_input,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_embeds_mask,
        timestep=timesteps / 1000,
        img_shapes=img_shapes,
        return_dict=False,
    )[0]
    
    # unpack
    model_pred = unpack_latents(
            model_pred, 
            height=model_input.shape[3],
            width=model_input.shape[4],
            vae_scale_factor=config.vae_scale_factor
        )

    weighting = compute_loss_weighting_for_sd3(weighting_scheme=config.weighting_scheme, sigmas=sigmas)
    
    target = noise - model_input
    
    # print('hii27')
    loss = (model_pred.float() - target.float()) ** 2
    
    loss = torch.mean((weighting.float() * loss).reshape(target.shape[0], -1), 1,)
    # print('hii31')
    bs = loss.shape[0]
    loss = loss.mean()
    # print('hii32')
    return loss, bs
    
    
    
@TRAIN_ITERATION_FUNCS.register_module()
def Qwen_Image_train_iteration_func(
    batch,
    vae,
    transformer,
    text_encoding_pipeline,
    noise_scheduler_copy,
    accelerator,
    config,
):  
    # ================= 1. Text Encoding (Online) =================
    # 从 batch 中获取文本列表
    prompts = batch["caption"]
    
    # 使用 pipeline 进行编码，注意加上 no_grad 节省显存
    with torch.no_grad():
        # 这里对应你预处理代码中的 text_pipeline.encode_prompt
        prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
            prompt=prompts, 
            max_sequence_length=config.max_sequence_length
        )
    
    # 确保 dtype 正确 (B, L, D)
    prompt_embeds = prompt_embeds.to(dtype=transformer.dtype, device=accelerator.device)
    prompt_embeds_mask = prompt_embeds_mask.to(device=accelerator.device)

    # ================= 2. Image Encoding (VAE Online) =================
    # 获取像素值 (B, 3, H, W)
    pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=vae.dtype)
    
    # Qwen-Image VAE 通常需要 5D 输入 (B, C, T, H, W)，如果是图片训练，T=1
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(2)
        
    bsz = pixel_values.shape[0]
    # 动态获取当前 batch 的 bucket 尺寸
    current_h, current_w = pixel_values.shape[3], pixel_values.shape[4]

    with torch.no_grad():
        # VAE Encode -> Sample
        dist = vae.encode(pixel_values).latent_dist
        latents = dist.sample()
    
    # ================= 3. Normalization & Noise =================
    # 标准化: (latents - mean) * std
    model_input = (latents - config.latents_mean) * config.latents_std
    model_input = model_input.to(dtype=transformer.dtype)

    # 生成噪声
    noise = torch.randn_like(model_input)
    
    # ================= 4. Timestep Sampling =================
    u = compute_density_for_timestep_sampling(
        weighting_scheme=config.weighting_scheme,
        batch_size=bsz,
        logit_mean=config.logit_mean,
        logit_std=config.logit_std,
        mode_scale=config.mode_scale,
    )
    
    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

    # ================= 5. Add Noise (Forward Process) =================
    with torch.no_grad():
        sigmas = get_sigmas(
            timesteps, 
            device=model_input.device, 
            noise_scheduler_copy=noise_scheduler_copy, 
            n_dim=model_input.ndim, 
            dtype=model_input.dtype
        )
        
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        
        # 构建 img_shapes 用于 transformer 的 patch 划分
        # 注意：这里假设 latents 已经下采样了 vae_scale_factor (通常是8)
        # transformer 内部可能还有 patch_size (通常是2)
        # 你的 parquet 代码逻辑是: bucket // scale // 2
        img_shapes = [
            (1, current_h // config.vae_scale_factor // 2, current_w // config.vae_scale_factor // 2)
        ] * bsz

    # ================= 6. Pack Latents =================
    # (B, C, T, H, W) -> (B, T, C, H, W) 或者是需要的特定 permute
    # 你的 parquet 代码中做了 permute(0, 2, 1, 3, 4)，即 (B, T, C, H, W)
    noisy_model_input = noisy_model_input.permute(0, 2, 1, 3, 4)
    
    packed_noisy_model_input = pack_latents(
        noisy_model_input,
        batch_size=bsz,
        num_channels_latents=model_input.shape[1],
        height=model_input.shape[3], # latent height
        width=model_input.shape[4],  # latent width
    )

    # ================= 7. Model Prediction =================
    model_pred = transformer(
        hidden_states=packed_noisy_model_input,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_embeds_mask,
        timestep=timesteps / 1000,
        img_shapes=img_shapes,
        return_dict=False,
    )[0]
    
    # ================= 8. Unpack & Loss =================
    model_pred = unpack_latents(
            model_pred, 
            height=model_input.shape[3],
            width=model_input.shape[4],
            vae_scale_factor=config.vae_scale_factor
        )

    weighting = compute_loss_weighting_for_sd3(weighting_scheme=config.weighting_scheme, sigmas=sigmas)
    
    target = noise - model_input
    
    # Loss 计算
    loss = (model_pred.float() - target.float()) ** 2
    loss = torch.mean((weighting.float() * loss).reshape(target.shape[0], -1), 1,)
    
    bs = loss.shape[0]
    loss = loss.mean()
    
    return loss, bs