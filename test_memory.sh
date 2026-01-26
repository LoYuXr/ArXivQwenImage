#!/bin/bash
# Quick test: Check if DeepSpeed can initialize with the model

set -e

echo "=========================================="
echo "Testing DeepSpeed ZeRO-3 Initialization"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux2

export HF_HOME="/home/v-yuxluo/data/huggingface_cache"
# Load HF_TOKEN from secrets file or environment
if [ -f ~/.hf_secrets ]; then
    source ~/.hf_secrets
fi
export PYTHONPATH="/home/v-yuxluo/WORK_local/ArXivQwenImage:$PYTHONPATH"

cd /home/v-yuxluo/WORK_local/ArXivQwenImage

# Test single GPU first to see memory usage
python -c "
import torch
from transformers import AutoTokenizer
from diffusers import Flux2Transformer2DModel

print('Loading Flux2Klein transformer...')
model = Flux2Transformer2DModel.from_pretrained(
    'black-forest-labs/FLUX.2-klein-base-9B',
    subfolder='transformer',
    torch_dtype=torch.bfloat16,
    token='$HF_TOKEN'
)
print(f'Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B')
print(f'GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')

# Move to GPU
model = model.to('cuda:0')
print(f'After moving to GPU: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')
print('✓ Model fits in single GPU for inference')
print('✗ But training needs ~108GB+ (model + gradients + optimizer states)')
"

echo ""
echo "DeepSpeed ZeRO-3 will split these across 4 GPUs:"
echo "  - Stage 3: Parameters + Gradients + Optimizer States sharded"
echo "  - Expected per-GPU: ~27GB (108GB / 4)"
echo "  - With activations: ~35-40GB per GPU"
echo ""
