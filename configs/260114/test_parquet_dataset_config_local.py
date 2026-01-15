_base_ = '../base_config.py'

# ====== 输入路径 ======
base_filesys_path = "/home/v-yuxluo/yuxuanluo"
# ====== 输出路径 ======
model_output_dir  = ''
image_output_dir  = ''
logging_dir       = 'logs'

# ====== 日志与监控 ======
report_to   = 'wandb'
tracker_name = 'test'
run_name = "yuxuan first try"
seed = 42

# Transformer 架构

# ====== 模型权重与结构 ======
transformer_cfg = dict(
    type='QwenImageTransformer2DModel',
)

# ====== LoRA ======
use_lora = True
lora_layers = "to_k,to_q,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
rank = 128
layer_weighting = 5.0


# ====== 训练设置 ======
train_batch_size = 4
gradient_accumulation_steps = 1
# resolution = 512
optimizer = "prodigy" #Adamw, muon, etc.
learning_rate = 1.0
lr_warmup_steps = 0
use_8bit_adam = False
max_train_steps = 100000
gradient_checkpointing = True
checkpointing_steps = 1000
resume_from_checkpoint = "latest"


# ====== 数据 ======
train_gpus_num = 4
dataloader_num_workers = 8 # --dataloader_num_workers
train_iteration_func_name = 'Qwen_Image_train_iteration_func_parquet' #'Qwen_Image_train_iteration_func' #Qwen_Image_train_iteration_func_parquet

use_parquet_dataset = True
dataset = dict(
        type='ArXiVParquetDataset',
        base_dir=base_filesys_path,
        parquet_base_path='ArXiV_parquet/QwenImageParquet_260115/',
        num_train_examples=502000,
        num_workers=dataloader_num_workers,
)


data_sampler = dict(
    type='DistributedBucketSampler',
    batch_size=train_batch_size,
    #num_replicas=accelerator.num_processes,  # <--- 总卡数
    #rank=accelerator.process_index,      # <--- 当前卡是第几号  #在后面再传
    drop_last=True,
    shuffle=True
)

max_sequence_length = 2048

# ====== 精度 / 性能 ======
mixed_precision = "bf16"


# ====== 验证与推理 ======
validation_func_name = 'Qwen_Image_train_iteration_func'
validation_steps = 50
num_inference_steps = 50
true_cfg_scale = 4.0
negative_prompt = " "

