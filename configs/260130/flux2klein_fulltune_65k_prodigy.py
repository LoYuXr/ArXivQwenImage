"""
Flux2Klein 9B Full Fine-tuning Config - Prodigy Optimizer
- optimizer = Prodigy (adaptive learning rate)
- lr_scheduler = constant_with_warmup (Prodigy handles lr internally)
- gradient_accumulation_steps = 2
- Effective batch = 1 × 2 × 4 GPUs = 8

Prodigy is an adaptive optimizer that automatically adjusts learning rate.
Recommended to use with constant/constant_with_warmup scheduler.

Dataset: ArXiV_parquet/Flux2Klein9BParquet_0128_save_h/
"""

_base_ = [
    '../base_config.py',
]

# Model Configuration
model_type = 'Flux2Klein'
base_filesys_path = "/mnt/data/"

model_dir = base_filesys_path + "models/Flux2Klein_9B"
model_path = model_dir
model_name_or_path = None

# ====== Model Weights ======
pretrained_model_name_or_path = "black-forest-labs/FLUX.2-klein-base-9B"
huggingface_token = None

transformer_cfg = dict(
    type='Flux2Transformer2DModel',
)

# Training Configuration
use_lora = False
train_batch_size = 1
gradient_accumulation_steps = 2

# ====== Dataset Configuration ======
use_parquet_dataset = True

dataset_cfg = dict(
    type='ArXiVParquetDatasetV3',
    base_dir=base_filesys_path,
    parquet_base_path='ArXiV_parquet/Flux2Klein9BParquet_0128_save_h',
    vae_scaling_factor=0.3611,
    num_workers=4,
    num_train_examples=None,
    debug_mode=False,
    is_main_process=True,
    stat_data=False,
)

sampler_cfg = dict(
    type='DistributedBucketSamplerV2',
    dataset=None,
    batch_size=1,
    num_replicas=1,
    rank=0,
    drop_last=True,
    shuffle=True,
)

# ====== Training Iteration Function ======
train_iteration_func = 'Flux2Klein_fulltune_train_iteration'

# Steps
max_train_steps = 65000
num_train_epochs = 1

checkpointing_steps = 500
checkpoints_total_limit = None
validation_steps = 500

# ====== EMA ======
use_ema = True
ema_decay = 0.9999
ema_update_after_step = 0
ema_steps = 100

# ====== Optimizer: Prodigy ======
# Prodigy is an adaptive optimizer - it adjusts LR automatically
# Recommended initial LR = 1.0 (Prodigy scales this internally)
optimizer = "prodigy"
learning_rate = 1.0  # Prodigy uses this as a base scale

# AdamW-compatible parameters (used by Prodigy)
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 0.01
adam_epsilon = 1e-8

# Prodigy-specific parameters
prodigy_beta3 = None  # If None, uses sqrt(adam_beta2)
prodigy_decouple = True  # Decouple weight decay (recommended)
prodigy_use_bias_correction = True
prodigy_safeguard_warmup = True  # Safeguard during warmup (recommended)

# ====== LR Scheduler ======
# For Prodigy, constant_with_warmup is recommended
# Prodigy handles adaptive LR internally
lr_scheduler = "constant_with_warmup"
lr_warmup_steps = 1000

# Mixed Precision
mixed_precision = "bf16"
allow_tf32 = True
gradient_checkpointing = True

# Dataset path
from_hf_hub = False
dataset_parquet_path = base_filesys_path + "ArXiV_parquet/Flux2Klein9BParquet_0128_save_h/"

# Data Processing
max_sequence_length = 1024 
dataloader_num_workers = 4
pin_memory = True

# Cache
cache_latents_to_disk = False
cache_text_embeddings_to_disk = False

# Validation
validation_func = 'Flux2Klein_fulltune_validation_func_parquet'
validation_guidance_scale = 3.5
num_inference_steps = 28

# Output
model_output_dir = base_filesys_path + "experiments/260130_fluxklein9B_prodigy"
output_dir = model_output_dir

# Logging
logging_steps = 10
logging_dir = model_output_dir + "/logs"
log_with = "wandb"
tracker_project_name = "0124OpenSciDraw_Flux2Klein_9B_Fulltune"
tracker_run_name = "260130_prodigy_adaptive"

# Resume
resume_from_checkpoint = "latest"

# Logging verbosity
verbose_logging = False

# WandB
report_to = "wandb"
wandb_project = "OpenSciDraw_Flux2Klein_9B_0124_Fulltune"

seed = 42

validation_prompts = [
    "The figure illustrates a process hooking mechanism using the LD_PRELOAD environment variable to inject a custom data collection library, siren.so, into target ELF binary executables during runtime. The global layout is a top-down flowchart depicting the sequence of interactions from environment setup to data analysis. At the top, a light blue rectangular box labeled 'Environment Variable: LD_PRELOAD=siren.so' initiates the process. This points downward to a green rectangle labeled 'Dynamic Linker: ld.so', which branches into two paths: one to a light blue box 'Injected Library: siren.so' and another to a green box 'Shared Libraries: DT_NEEDED'. Both converge into a large green rectangular container labeled 'ELF Binary Executable', which contains three internal components arranged vertically. The first is a light blue hexagon labeled 'Constructor: Data Collection and UDP Sender', followed by a green rectangle 'Application Code: main()', and then another light blue hexagon 'Destructor: Data Collection and UDP Sender'. These indicate that the injected library's data collection routines are triggered at both process startup (via constructor) and shutdown (via destructor). An arrow from the destructor leads to a light blue rectangle 'Message Receiver: UDP Server', which in turn connects to a light blue cylinder labeled 'Database: SQLite'. From the database, a downward arrow leads to a light blue rectangle 'Post-processing and Consolidation: Python', which then connects leftward to another light blue rectangle 'Statistics and Similarity Analysis: Python'. All elements shaded in light blue represent components of the SIREN architecture, while green elements denote standard system or application components. The arrows indicate the direction of control flow and data transmission, showing how injected data is sent via UDP, received, stored, processed, and finally analyzed. The diagram emphasizes the non-intrusive nature of the hooking mechanism, leveraging dynamic linking to collect runtime data without modifying the target application's source code.",
    "The figure presents an overview of four distinct end-to-end Task-Oriented Dialogue (TOD) approaches, arranged vertically as subfigures (a) through (d), each illustrating a different methodology for integrating language models into dialogue systems.",
    "The figure illustrates a network architecture for a single-step diffusion model with an enhanced decoder. The global layout is horizontal, progressing from left to right, with multiple parallel input streams converging into a central processing unit before diverging again toward the output.",
    "The figure presents a comparative diagram of four different defect detection tasks, labeled (a) ISDD, (b) MISDD, (c) MIISDD, and (d) MISDD-MM, illustrating variations in data modality handling and fusion strategies.",
    "The figure illustrates a model evaluation framework for a diffusion-based prediction system, structured as a horizontal workflow from left to right.",
    "The figure illustrates a linear probing framework applied to a frozen multimodal large language model (LLM) across different decoder layers, specifically focusing on the last-token representation at layer k.",
]

resolution_list = [
    [576, 960],
    [576, 960],
    [1008, 576],
    [1008, 576],
    [1008, 576],
    [1008, 576],
]
