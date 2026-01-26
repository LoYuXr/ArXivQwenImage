#!/bin/bash

# =========================================
# Flux2Klein Full Fine-tuning LOCAL Debug Script
# =========================================
# ⚠️ LOCAL VERSION - Not for AMLT submission
# Local path: /home/v-yuxluo/yuxuanluo = Remote: /mnt/data
# 
# This script launches full fine-tuning on 4x A100 GPUs
# Using conda environment: flux2
# 
# Usage:
#   bash run_fulltune_local.sh
# =========================================

set -e  # Exit on error

echo "=========================================="
echo "Flux2Klein Full Fine-tuning - Local Debug"
echo "=========================================="

# ====== Environment Setup ======
echo "[1/5] Setting up environment..."

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux2

# Set environment variables
export PYTHONPATH="/home/v-yuxluo/WORK_local/ArXivQwenImage:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=true
export HF_HOME="/home/v-yuxluo/data/huggingface_cache"  # Use cached models

# ====== Load tokens from secrets file if not already set ======
# Create a file ~/.hf_secrets with:
#   export HF_TOKEN="your_hf_token"
#   export WANDB_API_KEY="your_wandb_key"
if [ -f ~/.hf_secrets ]; then
    source ~/.hf_secrets
fi
# Or set them directly here (but don't commit to git!):
# export HF_TOKEN="${HF_TOKEN:-}"
# export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# WandB configuration
export WANDB_ENTITY=v-yuxluo
export WANDB_PROJECT=Flux2Klein-FullTune-Debug
export WANDB_BASE_URL="https://microsoft-research.wandb.io"
# export WANDB_API_KEY from ~/.hf_secrets

# NCCL configuration for multi-GPU training
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

echo "✓ Environment configured"
echo "  - Conda env: flux2"
echo "  - GPUs: 4x A100"
echo "  - Mixed precision: BF16"
echo ""

# ====== Check GPU Availability ======
echo "[2/5] Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ====== Verify Dataset ======
echo "[3/5] Verifying dataset..."
DATASET_PATH="/home/v-yuxluo/yuxuanluo/ArXiV_parquet/Flux2Klein9BParquet_260118"  # LOCAL Flux2Klein dataset
if [ -d "$DATASET_PATH" ]; then
    echo "✓ Dataset found at $DATASET_PATH"
    # Show directory structure
    echo "  Available years:"
    ls -1 "$DATASET_PATH" | head -5
else
    echo "✗ Error: Dataset not found at $DATASET_PATH"
    echo "  Expected local path: /home/v-yuxluo/yuxuanluo/ArXiV_parquet/Flux2Klein9BParquet_260118"
    echo "  (This equals /mnt/data/ArXiV_parquet/Flux2Klein9BParquet_260118 on remote AMLT machines)"
    exit 1
fi
echo ""

# ====== Create Output Directory ======
echo "[4/5] Creating output directory..."
OUTPUT_DIR="/home/v-yuxluo/WORK_local/ArXivQwenImage/output/flux2klein_fulltune_debug"
mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory: $OUTPUT_DIR"
echo ""

# ====== Launch Training ======
echo "[5/5] Launching training..."
echo "=========================================="
echo ""

cd /home/v-yuxluo/WORK_local/ArXivQwenImage

# Run training with accelerate + DeepSpeed ZeRO-3
# ZeRO-3 is REQUIRED for 9B full fine-tuning on 4x A100 80GB
accelerate launch \
    --config_file accelerate_cfg/deepspeed_zero3_bf16.yaml \
    train_OpenSciDraw_fulltune.py \
    configs/260121/flux2klein_fulltune_local_debug.py

echo ""
echo "=========================================="
echo "Training completed or interrupted"
echo "=========================================="
