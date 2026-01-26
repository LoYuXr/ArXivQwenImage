#!/usr/bin/env bash

# 1. 创建环境 (命名为 flux2)
conda create -n flux2 python=3.10 -y
conda activate flux2

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/huggingface/diffusers.git
pip install --upgrade transformers accelerate protobuf wandb
pip install opencv-python pandas pyarrow datasets ftfy sentencepiece einops scikit-learn timm mmengine tqdm
pip install bitsandbytes>=0.43.0 peft>=0.11.1 pydantic ray[train] prodigyopt

# 7. (强烈推荐) 安装 Flash Attention 2
# Flux 模型很大，Flash Attention 能显著减少显存占用并加速训练
# 这一步可能会编译较慢，需要系统中有 CUDA 编译器 (nvcc)
# pip install flash-attn --no-build-isolation
