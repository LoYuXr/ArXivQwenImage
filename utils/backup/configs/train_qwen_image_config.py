_base_ = './Qwen_base_config.py'

# ====== 核心路径配置 ======
# 你的 Qwen-Image 权重路径 (用于加载 Transformer)
pretrained_model_name_or_path = "Qwen/Qwen-Image-2512"

# 输出路径 (建议带上日期和分辨率标识)
model_output_dir  = '/mnt/data/ArXiV_parquet_dataset/qwen-768-latents-260114/'
logging_dir       = 'logs'

# ====== 日志与监控 ======
report_to    = 'wandb'
tracker_name = 'qwen-768-latents-260114-run1'
seed = 42

# # ====== 模型结构 ======
# transformer_cfg = dict(
#     type='QwenImageTransformer2DModel',
# )

# # ====== LoRA 设置 ======
# # 保持你之前的设置
# lora_layers = "to_k,to_q,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
# rank = 32
# layer_weighting = 5.0

# ====== 训练超参数 (关键修改) ======
# [修改点 1] Batch Size 可以改大了！
# 因为 Parquet 里存的是 Latents (16x96x96)，显存占用极小。
# A100 80G 可能能跑 32 甚至 64。建议先从 16 开始试。
# train_batch_size = 16 
# gradient_accumulation_steps = 1

# learning_rate = 2e-4
# optimizer = 'AdamW'
# use_8bit_adam = True 
# lr_scheduler = "constant"
# lr_warmup_steps = 1000 # 稍微给点 warmup
# max_train_steps = 100000 # 根据数据量调整
# gradient_checkpointing = True 
# checkpointing_steps = 5000
# resume_from_checkpoint = "latest"

# # 混合精度
# mixed_precision = "bf16" # A100 建议用 bf16，比 fp16 更稳

# # ====== 数据集配置 (核心修改) ======
# # [修改点 2] 替换为 Parquet Dataset
# # 注意：你需要确保代码里的 DATASETS 注册表能找到 'QwenFullCacheDataset'
# train_iteration_func_name = 'Qwen_Image_train_iteration_func' # 训练循环逻辑可能需要微调以适应 latents 输入

# dataset = dict(
#     type='QwenFullCacheDataset', # <--- 这里对应你在 registry 里注册的类名
    
#     # 指向具体的 Parquet 文件夹 (例如 2015 年的数据)
#     # 如果你想跑多年数据，可以让 Dataset 支持传入列表，或者把多年 parquet 放在同一个总目录下
#     parquet_folder="/mnt/data/ArXiV_parquet/768_pretrain_latents/2015",
    
#     # 这里的参数要和 QwenFullCacheDataset 的 __init__ 对应
#     # 比如是否需要传入 tokenizer 等（如果是 full cache 就不需要了）
# )

# # Dataloader 设置
# train_gpus_num = 1 # 根据实际提交的 GPU 数修改 (AMLT process_count_per_node)
# dataloader_num_workers = 8 

# # [修改点 3] 移除不需要的参数
# # resolution = 512  <-- 移除，因为 parquet 里已经是多尺度 bucket
# # max_sequence_length = 512 <-- 移除，因为 text embedding 已经 cache 好形状了

# # ====== VAE 配置 (核心修改) ======
# # [修改点 4] 移除 VAE Path
# # 因为我们直接读取 Latents，训练时根本不需要加载 VAE 模型！
# # multilayer_vae_path = ... <-- 移除或设为 None

# ====== 验证配置 ======
validation_func_name = 'QwenImage_multilayer_validation_func'
# 验证时需要 VAE 解码看图，所以验证函数内部可能需要临时加载 VAE，或者这里保留路径仅供验证用
validation_vae_path = "/mnt/data/models/Qwen-Image/vae" # 验证专用
validation_steps = 2000 # 跑得快了，验证频率可以降低
num_inference_steps = 50
true_cfg_scale = 4.0
negative_prompt = " "