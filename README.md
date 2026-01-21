# ScienceFlow

**ScienceFlow** is a unified training framework for scientific figure generation using diffusion models. It supports multiple model architectures (QwenImage, Flux2Klein, etc.) with a modular design for easy extension and experimentation.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Support](#model-support)
- [Configuration](#configuration)
- [Training](#training)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## ✨ Features

- **Multi-Model Support**: Unified interface for QwenImage, Flux2Klein, and more
- **Efficient Data Pipeline**: Parquet-based pre-computed latents and embeddings
- **LoRA Fine-tuning**: Memory-efficient training with comprehensive layer targeting
- **Flexible Training**: Support for various optimizers (AdamW, Prodigy), mixed precision, gradient accumulation
- **AMLT Integration**: Easy distributed training on Azure ML
- **Modular Design**: Easy to extend with new models, datasets, and training strategies

## 📁 Project Structure

```
ScienceFlow/
├── train_OpenSciDraw_loop.py    # Main training script
├── OpenSciDraw/                 # Core framework module
│   ├── datasets/                # Dataset implementations
│   ├── train_iteration_funcs/   # Model-specific training loops
│   ├── validation_funcs/        # Validation and inference
│   ├── utils/                   # Utilities (model factory, LoRA, etc.)
│   └── registry.py              # Component registry
├── configs/                     # Training configurations
│   ├── base_config.py          # Base configuration
│   └── 260120/                 # Experiment-specific configs
├── amlt/                       # Azure ML job configurations
├── env_setup/                  # Environment setup scripts
└── README.md                   # This file
```

See [OpenSciDraw/README.md](OpenSciDraw/README.md) for detailed module documentation.

## 🔧 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- Conda or virtualenv

### Environment Setup

```bash
# for qwenimage2512:
bash env_setup/init_qwen25_env.sh

# for flux.2-klein-base-9B
bash env_setup/init_flux2_env.sh
```

## 🚀 Quick Start

Please view `amlt/` for detailed runnning scripts. We use accelerate.

For any new experiments please modify `configs/base_config.py`

model_type = 'Flux2Klein'  # or 'QwenImage'
pretrained_model_name_or_path = "black-forest-labs/FLUX.2-klein-base-9B"

# Training settings
train_batch_size = 2
learning_rate = 1.0
optimizer = "prodigy"

# LoRA configuration
use_lora = True
rank = 128
lora_alpha = 128
lora_layers = "to_k,to_q,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,attn.to_out"

# Data
use_parquet_dataset = True
dataset = dict(
    type='ArXiVParquetDatasetV2',
    base_dir='/path/to/data/',
    parquet_base_path='ArXiV_parquet/Flux2Klein_latents/',
    num_train_examples=100000,
)
```

### 3. Train Locally

```bash
# Single GPU
accelerate launch train_OpenSciDraw_loop.py configs/my_experiment.py

# Multi-GPU (4 GPUs)
accelerate launch --config_file accelerate_cfg/1m4g_bf16.yaml \
    train_OpenSciDraw_loop.py configs/my_experiment.py
```

### 4. Train on Azure ML (AMLT)

```bash
# Submit job
amlt run -d <cluster_name> amlt/train_flux2klein.yaml
```

## 🎯 Model Support

### Flux2Klein (9B Parameters)

**Architecture:**
- **MMDiT Blocks** (8 layers): Dual-stream attention for image-text joint processing
- **Single Blocks** (24 layers): Parallel fused QKV+MLP for efficiency
- **VAE**: AutoencoderKLFlux2 with BatchNorm latent normalization
- **Text Encoder**: Qwen3-2.5B with 12288-dim output

**Key Features:**
- 4D Position IDs for RoPE: `[T, H, W, L]` format
- Flow matching with Euler discrete scheduler
- Patchify + BatchNorm normalization for latents
- Supports up to 1024 text sequence length

**Training Recommendations:**
```python
# Optimal LoRA configuration
rank = 128
lora_alpha = 128
lora_layers = "to_k,to_q,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,attn.to_out"

# Optimizer
optimizer = "prodigy"
learning_rate = 1.0

# Batch size (per GPU)
train_batch_size = 2  # A100 80GB
gradient_accumulation_steps = 1
```

### QwenImage (Legacy Support)

- 3D latent format
- Different position encoding
- See legacy documentation for details

## ⚙️ Configuration

### Key Configuration Parameters

```python
# Model
model_type = 'Flux2Klein'  # Model architecture
pretrained_model_name_or_path = "black-forest-labs/FLUX.2-klein-base-9B"

# Training
train_batch_size = 2
num_train_epochs = 2
max_train_steps = 50000
gradient_accumulation_steps = 1
mixed_precision = "bf16"
gradient_checkpointing = True

# Optimizer
optimizer = "prodigy"
learning_rate = 1.0
lr_warmup_steps = 0

# LoRA
use_lora = True
rank = 128
lora_alpha = 128
lora_dropout = 0.0
lora_layers = "to_k,to_q,to_v,..."

# Data
use_parquet_dataset = True
dataset = dict(
    type='ArXiVParquetDatasetV2',
    base_dir='/mnt/data/',
    parquet_base_path='ArXiV_parquet/...',
)

# Validation
validation_steps = 500
num_inference_steps = 28
guidance_scale = 3.5

# Checkpointing
checkpointing_steps = 1000
resume_from_checkpoint = "latest"

# Logging
report_to = 'wandb'
tracker_name = 'ScienceFlow-Experiment'
```

### LoRA Layer Targeting

**Flux2Klein Layer Breakdown:**

**MMDiT Blocks** (`transformer_blocks.*`):
- Attention: `to_k`, `to_q`, `to_v`, `to_out.0`, `add_k_proj`, `add_q_proj`, `add_v_proj`, `to_add_out`
- FeedForward: `ff.linear_in`, `ff.linear_out`
- FF Context: `ff_context.linear_in`, `ff_context.linear_out`

**Single Blocks** (`single_transformer_blocks.*`):
- Fused Attention: `attn.to_qkv_mlp_proj`, `attn.to_out`

**Recommended Configurations:**

```python
# Minimal (attention only)
lora_layers = "to_k,to_q,to_v,to_out.0"

# Standard (attention + cross-attention)
lora_layers = "to_k,to_q,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out"

# Full (attention + feedforward + single blocks) - RECOMMENDED
lora_layers = "to_k,to_q,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,attn.to_out"

# Explicit fal/ostris style (most verbose)
lora_layers = "attn.to_k,attn.to_q,attn.to_v,attn.to_out.0,attn.add_k_proj,attn.add_q_proj,attn.add_v_proj,attn.to_add_out,ff.linear_in,ff.linear_out,ff_context.linear_in,ff_context.linear_out,attn.to_qkv_mlp_proj,attn.to_out"
```

### LoRA Rank and Alpha

- **`rank`**: Dimension of LoRA matrices. Higher = more expressiveness, more parameters.
  - Small: 16-32 (subtle changes)
  - Medium: 64-128 (balanced)
  - Large: 256+ (high capacity)

- **`lora_alpha`**: Scaling factor for LoRA output. Effective strength = `alpha / rank`.
  - `alpha == rank`: Standard (1.0x strength)
  - `alpha < rank`: Reduced impact (e.g., 0.5x)
  - `alpha > rank`: Amplified impact (e.g., 2.0x)

**Recommendations:**
- Start with `rank = alpha` (e.g., 128/128)
- If overfitting: reduce alpha or rank
- If underfitting: increase rank

## 🏃 Training

### Local Training

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python train_OpenSciDraw_loop.py configs/my_config.py

# Multi-GPU with accelerate
accelerate launch --config_file accelerate_cfg/1m4g_bf16.yaml \
    train_OpenSciDraw_loop.py configs/my_config.py
```

### Azure ML (AMLT)

```bash
# Submit training job
amlt run -d <cluster_name> amlt/train_flux2klein.yaml -y

# Monitor job
amlt status <job_id>

# View logs
amlt logs <job_id>
```

### Resume Training

```python
# In config
resume_from_checkpoint = "latest"  # Auto-resume from latest checkpoint
# OR
resume_from_checkpoint = "checkpoint-5000"  # Resume from specific checkpoint
```

### Monitor Training

**WandB Integration:**
```python
report_to = 'wandb'
tracker_name = 'ScienceFlow-Flux2Klein'
run_name = "experiment_name"
```

**Key Metrics:**
- `loss`: Current step loss
- `train_loss`: Averaged loss over gradient accumulation
- `lr`: Current learning rate

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. **OOM (Out of Memory)**

**Symptoms:** `torch.cuda.OutOfMemoryError`

**Solutions:**
```python
# Reduce batch size
train_batch_size = 1

# Increase gradient accumulation
gradient_accumulation_steps = 4

# Enable gradient checkpointing
gradient_checkpointing = True

# Use lower precision
mixed_precision = "bf16"  # or "fp16"

# Reduce LoRA rank
rank = 64
```

## 🐛 Known Issues

1. **Flux2Klein VAE Loading**: Requires HuggingFace token for gated models
   - Solution: `huggingface-cli login` before training

2. **PEFT Version Compatibility**: Some PEFT versions have bugs with `add_adapter`
   - Recommended: `peft>=0.7.0`

3. **Mixed Precision on MPS**: BF16 not supported on Apple Silicon
   - Use `mixed_precision="fp16"` for Mac

## 📚 Additional Resources

- [Flux2Klein Official Repo](https://github.com/black-forest-labs/flux)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [PEFT Documentation](https://huggingface.co/docs/peft)

---

**Version:** 1.0.0  
**Last Updated:** January 2026  
**Maintainers:** [Your Team]
