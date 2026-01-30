#!/bin/bash
# ============================================================
# QwenImage 20B Full Fine-tuning Local Debug Script
# ============================================================
# 
# This script launches QwenImage 20B full fine-tuning with:
# - DeepSpeed ZeRO-3 for memory efficiency
# - Optimizer CPU offload to fit in 4x A100 80GB
# - BF16 mixed precision
#
# Prerequisites:
# 1. CUDA_VISIBLE_DEVICES should include 4 GPUs
# 2. QwenImage model cached in /home/v-yuxluo/data/huggingface_cache
# 3. Parquet dataset prepared in /home/v-yuxluo/data/ArXiV_parquet/qwenimage_latents
#
# Memory Requirements:
# - ~50-60GB GPU memory per card (4x A100 80GB recommended)
# - ~100GB+ CPU RAM for optimizer state offloading
#
# Expected Training Speed:
# - ~10-15 seconds per step with ZeRO-3 + CPU offload
# - Faster without CPU offload but may OOM
#
# ============================================================

set -e

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=true
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# HuggingFace token (if needed)
# export HF_TOKEN="your_token_here"

# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Change to project directory
cd /home/v-yuxluo/WORK_local/ArXivQwenImage

echo "============================================"
echo "Starting QwenImage 20B Full Fine-tuning"
echo "============================================"
echo "Model: QwenImage 20B"
echo "GPUs: 4x A100 80GB"
echo "Strategy: DeepSpeed ZeRO-3 + CPU Offload"
echo "Batch Size: 1 per GPU, GA=4"
echo "Effective Batch: 16"
echo "============================================"

# Launch training
accelerate launch \
    --config_file accelerate_cfg/deepspeed_zero3_qwenimage_20b.yaml \
    train_OpenSciDraw_fulltune.py \
    configs/260126/qwenimage_fulltune_local.py

echo "============================================"
echo "Training completed!"
echo "============================================"
