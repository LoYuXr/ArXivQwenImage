"""
Example training config for Flux2Klein 9B model
This demonstrates how to use the model factory to train a different model type.
"""

_base_ = ['../base_config.py', '../models/flux2klein_base.py']

# ====== Override Model Type ======
# This is set in flux2klein_base.py, but shown here for clarity
model_type = 'Flux2Klein'
pretrained_model_name_or_path = "black-forest-labs/Flux.2-Klein-9B"

# ====== Input/Output Paths ======
base_filesys_path = "/mnt/data"
model_output_dir = '/mnt/data/experiments/Flux2Klein/260118_test/models'
image_output_dir = '/mnt/data/experiments/Flux2Klein/260118_test/samples'
logging_dir = 'logs'

# ====== Logging & Monitoring ======
report_to = 'wandb'
tracker_name = 'Flux2Klein-Run'
run_name = "flux2klein 9B test"
seed = 42

# ====== Transformer Config ======
transformer_cfg = dict(
    type='Flux2Transformer2DModel',
)

# ====== LoRA ======
use_lora = True
lora_layers = "to_k,to_q,to_v,to_out.0"  # Default Flux2 layers
rank = 64
lora_alpha = 64
layer_weighting = 5.0

# ====== Training Settings ======
train_batch_size = 1
num_train_epochs = 2
gradient_accumulation_steps = 2
optimizer = "adamw"
learning_rate = 1e-4
lr_warmup_steps = 500
use_8bit_adam = False
max_train_steps = 50000
gradient_checkpointing = True
checkpointing_steps = 1000
resume_from_checkpoint = "latest"

# ====== Data ======
train_gpus_num = 4
dataloader_num_workers = 8
train_iteration_func_name = 'Flux2Klein_train_iteration_func'  # Would need to be implemented

use_parquet_dataset = True
dataset = dict(
    type='ArXiVParquetDatasetV2',
    base_dir=base_filesys_path,
    parquet_base_path='ArXiV_parquet/Flux2KleinParquet/',
    num_train_examples=500000,
    num_workers=dataloader_num_workers,
)

data_sampler = dict(
    type='ArXiVMixScaleBatchSampler',
    batch_size=train_batch_size,
)

max_sequence_length = 512

# ====== Precision / Performance ======
mixed_precision = "bf16"

# ====== Validation & Inference ======
validation_func_name = 'Flux2Klein_validation_func'  # Would need to be implemented
validation_steps = 500
num_inference_steps = 28
true_cfg_scale = 3.5
negative_prompt = ""

validation_prompts = [
    "A detailed scientific diagram showing the structure of a neuron...",
    "A flowchart illustrating the machine learning training pipeline...",
]

resolution_list = [
    (1024, 1024),
    (768, 1024),
]
