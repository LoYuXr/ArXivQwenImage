# QwenImage 20B Full Fine-tuning 技术文档

## 概述

本文档描述了如何对 QwenImage-2512 (约20B参数) diffusion transformer 进行全参数微调。由于模型参数量较大，我们使用 **DeepSpeed ZeRO-3 with CPU Offload** 策略进行分布式训练。

**🌟 经过验证的成功方案**: 使用 `DS_SKIP_CUDA_CHECK=1` + DeepSpeed ZeRO-3 + CPU Optimizer Offload

## 模型规格

| 属性 | 值 |
|------|-----|
| 模型名称 | Qwen/Qwen-Image-2512 |
| Transformer 参数量 | ~20B |
| 注意力层数 | 60 |
| 注意力头数 | 24 |
| 注意力头维度 | 128 |
| 输入通道数 | 64 |
| 输出通道数 | 16 |

## 内存估算

对于 20B 参数模型的全参数微调：

| 组件 | 无分片 | ZeRO-3 (4 GPU) + CPU Offload |
|------|--------|------------------------------|
| 模型参数 (BF16) | 40 GB | 10 GB/GPU |
| 梯度 (BF16) | 40 GB | 10 GB/GPU (CPU) |
| 优化器状态 (AdamW) | 160 GB | CPU (约200GB) |
| **GPU 总计** | **~240 GB** | **~20 GB/GPU** |
| **CPU 总计** | - | **~200 GB** |

实际测试：4x A100 80GB 上每个GPU使用约20GB，CPU使用约225GB。

## 文件结构

```
ArXivQwenImage/
├── accelerate_cfg/
│   ├── deepspeed_zero3_qwenimage_20b.yaml       # ✅ 推荐: ZeRO-3 + CPU offload
│   ├── fsdp_qwenimage_cpu_offload.yaml          # FSDP (有dtype问题)
│   ├── fsdp_qwenimage_pure_bf16.yaml            # FSDP 纯GPU (OOM)
│   └── deepspeed_zero3_qwenimage_pure_gpu.yaml  # ZeRO-3 纯GPU (OOM)
├── configs/260127/
│   └── qwenimage_fulltune_5000.py               # 5000步训练配置
├── configs/260126/
│   ├── qwenimage_fulltune_local_debug.py        # 本地调试配置
│   ├── qwenimage_fulltune_local.py              # 本地训练配置
│   └── qwenimage_fulltune_amlt.py               # AMLT生产配置
├── OpenSciDraw/train_iteration_funcs/
│   └── QwenImage_fulltune_iteration_func.py     # 训练迭代函数
├── OpenSciDraw/validation_funcs/
│   └── QwenImage_fulltune_validation_func.py    # 验证函数
├── train_OpenSciDraw_fulltune.py                # 主训练脚本
└── run_qwenimage_20b_fulltune.sh                # ✅ 启动脚本
```

## 训练命令

### 🌟 推荐: DeepSpeed ZeRO-3 with CPU Offload (本地 4x A100 80GB)

**重要**: 由于系统CUDA版本(11.8)与PyTorch CUDA(12.1)不匹配，需要设置 `DS_SKIP_CUDA_CHECK=1`

```bash
# 使用启动脚本（已包含所有正确设置）
./run_qwenimage_20b_fulltune.sh

# 或手动运行
DS_SKIP_CUDA_CHECK=1 accelerate launch \
    --config_file accelerate_cfg/deepspeed_zero3_qwenimage_20b.yaml \
    train_OpenSciDraw_fulltune.py \
    configs/260127/qwenimage_fulltune_5000.py
```

### ⚠️ 不推荐的配置

#### FSDP with CPU Offload
```bash
# ❌ 有 gradient dtype 不匹配问题 (bf16 vs float32)
accelerate launch \
    --config_file accelerate_cfg/fsdp_qwenimage_cpu_offload.yaml \
    train_OpenSciDraw_fulltune.py \
    configs/260127/qwenimage_fulltune_5000.py
```

#### DeepSpeed/FSDP 纯GPU
```bash
# ❌ 4x A100 80GB 内存不足 (OOM in optimizer.step)
accelerate launch \
    --config_file accelerate_cfg/deepspeed_zero3_qwenimage_pure_gpu.yaml \
    train_OpenSciDraw_fulltune.py \
    configs/260127/qwenimage_fulltune_5000.py
```

## 配置说明

### DeepSpeed ZeRO-3 配置 (`deepspeed_zero3_qwenimage_20b.yaml`) ✅ 推荐

```yaml
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: true          # CPU offload
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
```

### DeepSpeed ZeRO-3 配置 (`deepspeed_zero3_qwenimage_pure_gpu.yaml`)

```yaml
deepspeed_config:
  zero_stage: 3                    # ZeRO-3: 完全分片
  zero3_init_flag: true            # ZeRO-3初始化
  zero3_save_16bit_model: true     # 保存16bit模型
  gradient_accumulation_steps: 4   # 梯度累积
  gradient_clipping: 1.0           # 梯度裁剪
  train_micro_batch_size_per_gpu: 1
  offload_optimizer_device: none   # 纯GPU
  offload_param_device: none
```

### 训练配置 (`qwenimage_fulltune_local_debug.py`)

关键参数:
- `model_type = 'QwenImage'`: 指定模型类型
- `train_iteration_func = 'QwenImage_fulltune_train_iteration'`: 训练迭代函数
- `validation_func = 'QwenImage_fulltune_validation_func_parquet'`: 验证函数(自动选择)
- `use_lora = False`: 关闭LoRA，进行全参数微调
- `gradient_checkpointing = True`: 启用梯度检查点节省内存
- `learning_rate = 5e-6`: 较低的学习率用于大模型

## 与 Flux2Klein 的对比

| 特性 | Flux2Klein 9B | QwenImage 20B |
|------|---------------|---------------|
| 参数量 | ~9B | ~20B |
| 推荐策略 | DeepSpeed ZeRO-2 | **FSDP with CPU offload** |
| Latent格式 | 4D (B,C,H,W) | 5D (B,C,T,H,W) |
| 位置编码 | img_ids + txt_ids | img_shapes |
| 训练速度 | ~5s/step | ~95-105s/step |
| 迭代函数 | Flux2Klein_fulltune_train_iteration | QwenImage_fulltune_train_iteration |
| 验证函数 | Flux2Klein_fulltune_validation_func_parquet | QwenImage_fulltune_validation_func_parquet |

## 训练监控

训练过程中会输出以下日志:
- Loss 值 (正常范围: 0.2-1.5)
- 学习率
- Latent 统计信息 (mean, std)

使用 WandB 进行可视化监控:
```python
report_to = "wandb"
wandb_project = "QwenImage-20B-FullTune"
```

## 故障排除

### 1. DeepSpeed CUDA 版本不匹配
错误: `CUDAMismatchException: Installed CUDA version 11.8 does not match the version torch was compiled with 12.1`

**解决方案**: 使用 FSDP 代替 DeepSpeed
```bash
accelerate launch --config_file accelerate_cfg/fsdp_qwenimage_cpu_offload.yaml ...
```

### 2. CUDA OOM
- 减小 `train_batch_size` 为 1
- 增加 `gradient_accumulation_steps`
- 使用 FSDP with CPU offload

### 3. CPU 内存不足
- 确保有足够CPU内存 (建议 >= 100GB)
- 减少 `dataloader_num_workers`

### 4. 训练速度过慢
- 检查 CPU offload 是否必要
- 增加 GPU 数量
- 使用 NVMe offload (需要额外配置)

### 5. Loss 不下降
- 检查学习率是否合适 (1e-6 到 1e-5)
- 确认数据加载正确
- 验证 latent 归一化是否正确

### 6. 验证函数报错 'FrozenDict' object has no attribute ...
**解决方案**: 使用 QwenImage 专用验证函数
```python
# 在配置中设置 (或自动选择):
validation_func = 'QwenImage_fulltune_validation_func_parquet'
```

## 性能预期

在 4x A100 80GB 上:
- 使用 FSDP + CPU offload: ~95-105s/step
- 有效batch size: 8 (1 x 2 x 4 GPUs)
- 50 步约需: 1.5-2 小时
- 5000 步约需: 5-6 天

## 已验证的成功配置

✅ **本地验证通过** (2026-01-26):
- 环境: flux2 (PyTorch 2.5.1+cu121)
- 配置: `fsdp_qwenimage_cpu_offload.yaml`
- 训练配置: `qwenimage_fulltune_local_debug.py`
- 结果: 50步完成，loss正常下降 (0.017 → 0.007)
- WandB: https://microsoft-research.wandb.io/v-yuxluo/QwenImage-20B-Debug

## 后续步骤

1. 首先运行本地调试 (1000 步) 验证训练正常
2. 检查 loss 曲线和生成样本质量
3. 调整超参数后在 AMLT 上进行完整训练
