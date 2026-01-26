#!/bin/bash
# Flux2Klein Full Fine-tuning with DeepSpeed ZeRO-3 - 5000 Steps
set -e

source /home/v-yuxluo/miniconda3/etc/profile.d/conda.sh
conda activate flux2
cd /home/v-yuxluo/WORK_local/ArXivQwenImage

export HF_HOME=/home/v-yuxluo/data/huggingface_cache
export WANDB_MODE=online
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=========================================="
echo "Starting Flux2Klein Full Fine-tuning (5000 steps)"
echo "Using DeepSpeed ZeRO-2 with 4 GPUs"
echo "=========================================="
date

accelerate launch \
    --config_file accelerate_cfg/deepspeed_zero2_bf16.yaml \
    --num_processes 4 \
    train_OpenSciDraw_fulltune.py \
    configs/260122/flux2klein_fulltune_5000.py

echo "=========================================="
echo "Training completed!"
echo "=========================================="
date
