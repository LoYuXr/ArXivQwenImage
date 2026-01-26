"""
Flux2Klein 9B Full Fine-tuning Config for Azure ML (AMLT)
- 50000 steps
- DeepSpeed ZeRO-2
- AdamW optimizer with lr=1e-5
- 400K training samples
"""

# Model Configuration
model_type = 'Flux2Klein'
base_filesys_path = "/mnt/data/"

model_dir = base_filesys_path + "models/Flux2Klein_9B"
model_path = model_dir
# For safety, we don't load the weights at training start; transformer weights are loaded separately
model_name_or_path = None

# Training Configuration
use_lora = False  # Full fine-tuning
train_batch_size = 1
gradient_accumulation_steps = 4  # Effective batch = 1 * 4 * 4 GPUs = 16

# Steps
max_train_steps = 50000
num_train_epochs = 1  # Will be capped by max_train_steps
checkpointing_steps = 100
checkpoints_total_limit = 3  # Keep only 3 checkpoints (~340GB)
validation_steps = 100

# Learning Rate
learning_rate = 1e-5
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 0.01
adam_epsilon = 1e-8

# Warmup - ~2% of total steps for 400K samples
lr_warmup_steps = 1000
lr_scheduler = "cosine"  # Cosine annealing after warmup

# Mixed Precision
mixed_precision = "bf16"
allow_tf32 = True

# Dataset - Parquet format
from_hf_hub = False
dataset_parquet_path = base_filesys_path + "ArXiV_parquet/Flux2Klein9BParquet_260118/"

# Data Processing
max_sequence_length = 512
dataloader_num_workers = 4
pin_memory = True

# Cache
cache_latents_to_disk = False  # Already pre-computed in parquet
cache_text_embeddings_to_disk = False

# Validation
num_validation_images = 4

# Output
model_output_dir = base_filesys_path + "experiments/260122_fluxklein9Bbase_fullfinetune_Adamw_1e_5"
output_dir = model_output_dir

# Logging
logging_steps = 10
logging_dir = model_output_dir + "/logs"
log_with = "wandb"
tracker_project_name = "OpenSciDraw_Flux2Klein_9B_Fulltune"
tracker_run_name = "260122_50k_steps_adamw_1e-5"

# Resume
resume_from_checkpoint = None  # Set to "latest" to resume from last checkpoint

# Debug
report_to = "wandb"
seed = 42
