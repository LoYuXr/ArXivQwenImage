# QwenImage 20B Full Fine-tuning Configuration
# 5000 steps local training with FSDP

# =========================================================
# Model Configuration
# =========================================================
seed = 42
device = "cuda"
dtype = "float32"
revision = None
variant = None
bnb_quantization_config_path = None

# Model type - determines which model components to load
model_type = "QwenImage"

# Transformer configuration
transformer_cfg = {
    "type": "QwenImageTransformer2DModel",
}

# Model paths
pretrained_model_name_or_path = "Qwen/Qwen-Image-2512"
huggingface_token = None

# =========================================================
# Training Mode: Full Fine-tuning
# =========================================================
use_lora = False  # Full fine-tuning, not LoRA

# LoRA parameters (not used when use_lora=False, but kept for compatibility)
lora_layers = "to_k,to_q,to_v"
rank = 64
lora_alpha = 4
lora_dropout = 0.0
layer_weighting = 5.0
pos_embedding = "rope"
decoder_arch = "vit"

# =========================================================
# Dataset Configuration
# =========================================================
use_parquet_dataset = True

# Training parameters
train_batch_size = 1
num_train_epochs = 100  # Will be limited by max_train_steps
max_train_steps = 5000
gradient_accumulation_steps = 4  # Increased to reduce memory per step
gradient_checkpointing = True
cache_latents = False

# =========================================================
# Optimizer Configuration
# =========================================================
optimizer = "adamw"
use_8bit_adam = False
learning_rate = 5e-6
lr_scheduler = "cosine"
lr_warmup_steps = 100

# AdamW parameters
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 0.01
adam_epsilon = 1e-08
max_grad_norm = 1.0

# Prodigy parameters (not used with adamw)
prodigy_beta3 = None
prodigy_decouple = True
prodigy_use_bias_correction = True
prodigy_safeguard_warmup = True

# =========================================================
# Checkpointing
# =========================================================
checkpointing_steps = 500
resume_from_checkpoint = "latest"
checkpoints_total_limit = None  # Changed from 5 to keep all checkpoints

# =========================================================
# Mixed Precision & Hardware
# =========================================================
mixed_precision = "bf16"
allow_tf32 = True
upcast_before_saving = False
offload = False

# =========================================================
# Logging & Tracking
# =========================================================
report_to = "wandb"
push_to_hub = False
hub_token = None
hub_model_id = None

# Cache directory
cache_dir = "/home/v-yuxluo/data/huggingface_cache"

# Learning rate scaling
scale_lr = False
lr_num_cycles = 1
lr_power = 1.0

# =========================================================
# Flow Matching Configuration
# =========================================================
weighting_scheme = "logit_normal"
logit_mean = 0.0
logit_std = 1.0
mode_scale = 1.29
validation_guidance_scale = 1.0

# =========================================================
# Dataset - Parquet Configuration
# =========================================================
base_filesys_path = "/home/v-yuxluo/data/"

dataset_cfg = {
    "type": "ArXiVParquetDatasetV2",
    "base_dir": "/home/v-yuxluo/data/",
    "parquet_base_path": "ArXiV_parquet/qwenimage_latents",
    "num_workers": 2,
    "num_train_examples": None,  # Use all data
    "debug_mode": False,
    "is_main_process": True,
    "stat_data": False,
}

sampler_cfg = {
    "type": "DistributedBucketSamplerV2",
    "dataset": None,
    "batch_size": 1,
    "num_replicas": None,
    "rank": None,
    "drop_last": True,
    "shuffle": True,
}

# Training iteration function - QwenImage specific
train_iteration_func = "QwenImage_fulltune_train_iteration"

# =========================================================
# Validation Configuration
# =========================================================
# Note: FSDP mode has issues with validation during training
# Setting to very high value to skip real-time validation
# Validation can be done manually after checkpoints are saved
validation_steps = 99999  # Effectively disable validation during FSDP training
max_sequence_length = 1024
dataloader_num_workers = 2
pin_memory = True
num_inference_steps = 50

# =========================================================
# Output Configuration
# =========================================================
model_output_dir = "/home/v-yuxluo/data/experiments/260127_qwenimage_20b_5000steps"
output_dir = "/home/v-yuxluo/data/experiments/260127_qwenimage_20b_5000steps"
log_steps = 10
verbose_logging = True
logging_dir = "/home/v-yuxluo/data/experiments/260127_qwenimage_20b_5000steps/logs"

# WandB configuration
wandb_project = "QwenImage-20B-FullTune"
tracker_run_name = "260127_qwenimage_5000steps"

# Validation prompts
validation_prompts = [
    "A scientific diagram showing a neural network architecture with multiple layers",
    "A chart comparing machine learning algorithms performance",
]

resolution_list = [
    [768, 768],
    [768, 768],
]
