#!/usr/bin/env bash

conda create -n qwen25 python=3.10 -y; conda activate qwen25
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/huggingface/diffusers
pip install transformers==4.55.0 accelerate protobuf wandb #4.51.3不支持dict的config传入
pip install opencv-python pandas pyarrow datasets ftfy sentencepiece einops scikit-learn timm==0.9.2 mmengine tqdm
pip install bitsandbytes>=0.43.0 peft pydantic ray[train] prodigyopt


# 测试 CUDA 是否可用
python -c "import torch; print('CUDA is available:', torch.cuda.is_available())"