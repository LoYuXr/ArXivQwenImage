"""
QwenImage 20B Full Fine-tuning Config - Azure ML (AMLT) Version
For production training on Azure ML with 4x A100 80GB GPUs

Key configurations:
- model_type: QwenImage (20B transformer)
- use_parquet_dataset: True (pre-computed latents/embeddings)
- DeepSpeed ZeRO-3 with optimizer CPU offload
- gradient_accumulation_steps: 4
- Effective batch = 1 × 4 × 4 GPUs = 16
- 50000 max training steps

Memory Estimation for 20B model with ZeRO-3:
- Model params sharded: 20B / 4 GPUs * 2 bytes = 10GB per GPU
- Gradients sharded: 10GB per GPU
- Optimizer states (offloaded to CPU): ~0GB GPU
- Activations + buffers: ~30-40GB per GPU (with gradient checkpointing)
- Total GPU usage: ~50-60GB per GPU (fits in 80GB A100)

Usage:
    accelerate launch --config_file accelerate_cfg/deepspeed_zero3_qwenimage_20b.yaml \\
        train_OpenSciDraw_fulltune.py configs/260126/qwenimage_fulltune_amlt.py
"""

_base_ = [
    '../base_config.py',
]

# ====== Model Configuration ======
model_type = 'QwenImage'
pretrained_model_name_or_path = "Qwen/Qwen-Image-2512"

# AMLT: No local cache, use HF Hub directly
cache_dir = None

# HuggingFace token is managed via environment variable HF_TOKEN
huggingface_token = None

transformer_cfg = dict(
    type='QwenImageTransformer2DModel',
)

# ====== Training Configuration ======
use_lora = False  # Full fine-tuning - no LoRA
train_batch_size = 1
gradient_accumulation_steps = 4  # Effective batch = 1 * 4 * 4 GPUs = 16

# ====== File System Paths (AMLT) ======
base_filesys_path = "/mnt/data/"

# ====== Dataset Configuration ======
use_parquet_dataset = True

dataset_cfg = dict(
    type='ArXiVParquetDatasetV2',
    base_dir=base_filesys_path,
    parquet_base_path='ArXiV_parquet/QwenImage2512Parquet',  # Production parquet path
    num_workers=4,
    num_train_examples=None,  # Use all available data (~500K samples)
    debug_mode=False,
    is_main_process=True,
    stat_data=False,
)

sampler_cfg = dict(
    type='DistributedBucketSamplerV2',
    dataset=None,
    batch_size=1,
    num_replicas=None,
    rank=None,
    drop_last=True,
    shuffle=True,
)

# ====== Training Iteration Function ======
train_iteration_func = 'QwenImage_fulltune_train_iteration'

# ====== Training Steps ======
# Full training: 50000 steps
# With GA=4 and 4 GPUs, effective batch = 16
# Total samples seen = 50000 * 16 = 800K samples
max_train_steps = 50000
num_train_epochs = 2

# Checkpointing
checkpointing_steps = 500
checkpoints_total_limit = None  # Changed from 3 to keep all checkpoints
validation_steps = 500

# ====== Learning Rate ======
learning_rate = 1e-5  # Standard for large model fine-tuning
optimizer = "adamw"
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 0.01
adam_epsilon = 1e-8

# Warmup - ~1% of total steps
lr_warmup_steps = 500
lr_scheduler = "cosine"
lr_num_cycles = 1
lr_power = 1.0

# ====== Mixed Precision ======
mixed_precision = "bf16"
allow_tf32 = True

# ====== Gradient Settings ======
max_grad_norm = 1.0
gradient_checkpointing = True

# ====== Data Processing ======
max_sequence_length = 1024
dataloader_num_workers = 4
pin_memory = True

# ====== Flow Matching Settings ======
weighting_scheme = "logit_normal"
logit_mean = 0.0
logit_std = 1.0
mode_scale = 1.29

# ====== Validation ======
validation_guidance_scale = 1.0
num_inference_steps = 50

# ====== Output ======
model_output_dir = base_filesys_path + "experiments/260126_qwenimage_20b_fulltune"
output_dir = model_output_dir

# ====== Logging ======
log_steps = 10
verbose_logging = False  # Reduce log verbosity in production
logging_dir = model_output_dir + "/logs"
report_to = "wandb"
wandb_project = "QwenImage-20B-FullTune"
tracker_run_name = "260126_qwenimage_20b_fulltune_amlt"

# ====== Resume ======
resume_from_checkpoint = "latest"

# ====== Validation Prompts ======
validation_prompts = [
    "The figure illustrates a neural network architecture with multiple layers connected by arrows.",
    "A bar chart showing the comparison of model performance across different datasets.",
    "A flowchart depicting the data processing pipeline from input to output.",
    "The diagram shows a transformer architecture with self-attention mechanism.",
]

resolution_list = [
    [768, 768],
    [768, 768],
    [768, 768],
    [768, 768],
]
