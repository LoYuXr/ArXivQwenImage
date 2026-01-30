#!/bin/bash
# QwenImage 20B Full Fine-tuning Training Script
# Uses DeepSpeed ZeRO-3 with CPU offload optimizer
# 
# Hardware: 4x NVIDIA A100 80GB
# Model: Qwen/Qwen-Image-2512 (20B parameters)
# Training: 5000 steps, batch_size=1, gradient_accumulation=4
#
# IMPORTANT: DS_SKIP_CUDA_CHECK=1 is required due to CUDA version mismatch
# (System CUDA 11.8 vs PyTorch CUDA 12.1)

set -e

cd /home/v-yuxluo/WORK_local/ArXivQwenImage

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux2

# Create logs directory
mkdir -p logs

# Set timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/qwenimage_20b_fulltune_${TIMESTAMP}.log"

echo "Starting QwenImage 20B Full Fine-tuning..."
echo "Log file: ${LOG_FILE}"
echo ""

# Run training with DS_SKIP_CUDA_CHECK to avoid CUDA version mismatch
DS_SKIP_CUDA_CHECK=1 nohup accelerate launch \
    --config_file accelerate_cfg/deepspeed_zero3_qwenimage_20b.yaml \
    train_OpenSciDraw_fulltune.py \
    configs/260127/qwenimage_fulltune_5000.py \
    > "${LOG_FILE}" 2>&1 &

echo "Training started with PID: $!"
echo ""
echo "Monitor with:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Check progress with:"
echo "  grep -o 'Steps:.*it]' ${LOG_FILE} | tail -3"
echo ""
echo "Check GPU usage with:"
echo "  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv"
