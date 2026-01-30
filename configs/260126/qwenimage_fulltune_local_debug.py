"""
QwenImage 20B Full Fine-tuning Config - Local Debug (Quick Test)
For quick validation on local 4x A100 80GB GPUs with DeepSpeed ZeRO-3

This is a minimal debug config with only 50 steps for fast testing.

Usage:
    conda activate qwen25
    accelerate launch --config_file accelerate_cfg/deepspeed_zero3_qwenimage_20b.yaml \\
        train_OpenSciDraw_fulltune.py configs/260126/qwenimage_fulltune_local_debug.py
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
gradient_accumulation_steps = 2  # Smaller GA for faster debug

# ====== File System Paths ======
# Use local path, NOT blob path (faster)
base_filesys_path = "/home/v-yuxluo/data/"

# ====== Dataset Configuration ======
use_parquet_dataset = True

dataset_cfg = dict(
    type='ArXiVParquetDatasetV2',
    base_dir=base_filesys_path,
    parquet_base_path='ArXiV_parquet/qwenimage_latents',
    num_workers=2,
    num_train_examples=100,  # Only 100 samples for quick debug
    debug_mode=True,
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
max_train_steps = 50  # Very short for debug
num_train_epochs = 1

# Checkpointing - less frequent for debug
checkpointing_steps = 25
checkpoints_total_limit = None  # Changed from 2 to keep all checkpoints
validation_steps = 25

# ====== Learning Rate ======
learning_rate = 5e-6
optimizer = "adamw"
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 0.01
adam_epsilon = 1e-8

# Warmup
lr_warmup_steps = 10
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
dataloader_num_workers = 2
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
model_output_dir = "/home/v-yuxluo/data/experiments/260126_qwenimage_20b_debug"
output_dir = model_output_dir

# ====== Logging ======
log_steps = 5
verbose_logging = True  # Enable detailed debug output
logging_dir = model_output_dir + "/logs"
report_to = "wandb"
wandb_project = "QwenImage-20B-Debug"
tracker_run_name = "260126_qwenimage_debug_test"

# ====== Resume ======
resume_from_checkpoint = "latest"

# ====== Validation Prompts (minimal for debug) ======
validation_prompts = [
    "The figure illustrates a neural network architecture.",
]

resolution_list = [
    [768, 768],
]
