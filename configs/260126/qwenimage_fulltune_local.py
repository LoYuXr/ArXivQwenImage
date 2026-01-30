"""
QwenImage 20B Full Fine-tuning Config - Local Debug Version
For testing on local 4x A100 80GB GPUs with DeepSpeed ZeRO-3

Key configurations:
- model_type: QwenImage (20B transformer)
- use_parquet_dataset: True (pre-computed latents/embeddings)
- DeepSpeed ZeRO-3 with optimizer CPU offload
- gradient_accumulation_steps: 4
- Effective batch = 1 × 4 × 4 GPUs = 16

Usage:
    accelerate launch --config_file accelerate_cfg/deepspeed_zero3_qwenimage_20b.yaml \\
        train_OpenSciDraw_fulltune.py configs/260126/qwenimage_fulltune_local.py
"""

_base_ = [
    '../base_config.py',
]

# ====== Model Configuration ======
model_type = 'QwenImage'
pretrained_model_name_or_path = "Qwen/Qwen-Image-2512"

# Local cache dir
cache_dir = "/home/v-yuxluo/data/huggingface_cache"

# HuggingFace token is managed centrally
huggingface_token = None

transformer_cfg = dict(
    type='QwenImageTransformer2DModel',
)

# ====== Training Configuration ======
use_lora = False  # Full fine-tuning - no LoRA
train_batch_size = 1
gradient_accumulation_steps = 4  # Effective batch = 1 * 4 * 4 GPUs = 16

# ====== File System Paths ======
base_filesys_path = "/home/v-yuxluo/data/"

# ====== Dataset Configuration ======
use_parquet_dataset = True

dataset_cfg = dict(
    type='ArXiVParquetDatasetV2',
    base_dir=base_filesys_path,
    parquet_base_path='ArXiV_parquet/qwenimage_latents',
    num_workers=4,
    num_train_examples=None,  # Use all available data
    debug_mode=False,
    is_main_process=True,
    stat_data=False,
)

sampler_cfg = dict(
    type='DistributedBucketSamplerV2',
    dataset=None,  # Filled in training script
    batch_size=1,
    num_replicas=None,  # Filled in training script
    rank=None,  # Filled in training script
    drop_last=True,
    shuffle=True,
)

# ====== Training Iteration Function ======
train_iteration_func = 'QwenImage_fulltune_train_iteration'

# ====== Training Steps ======
# For local debug: start with fewer steps
max_train_steps = 1000  # Debug: reduce for testing
num_train_epochs = 1

# Checkpointing
checkpointing_steps = 100
checkpoints_total_limit = None  # Changed from 3 to keep all checkpoints
validation_steps = 100

# ====== Learning Rate ======
# Lower LR for full fine-tuning of large model
learning_rate = 5e-6  # Conservative for 20B model
optimizer = "adamw"
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 0.01
adam_epsilon = 1e-8

# Warmup
lr_warmup_steps = 100
lr_scheduler = "cosine"
lr_num_cycles = 1
lr_power = 1.0

# ====== Mixed Precision ======
mixed_precision = "bf16"
allow_tf32 = True

# ====== Gradient Settings ======
max_grad_norm = 1.0
gradient_checkpointing = True  # Critical for memory efficiency

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
validation_guidance_scale = 1.0  # For training, usually 1.0
num_inference_steps = 50

# ====== Output ======
model_output_dir = base_filesys_path + "experiments/260126_qwenimage_20b_fulltune_local"
output_dir = model_output_dir

# ====== Logging ======
log_steps = 10
verbose_logging = True  # Enable detailed debug output
logging_dir = model_output_dir + "/logs"
report_to = "wandb"
wandb_project = "QwenImage-20B-FullTune"
tracker_run_name = "260126_qwenimage_20b_fulltune_local_debug"

# ====== Resume ======
resume_from_checkpoint = "latest"

# ====== Validation Prompts (for monitoring) ======
validation_prompts = [
    "The figure illustrates a neural network architecture with multiple layers connected by arrows.",
    "A bar chart showing the comparison of model performance across different datasets.",
    "A flowchart depicting the data processing pipeline from input to output.",
]

resolution_list = [
    [768, 768],
    [768, 768],
    [768, 768],
]
