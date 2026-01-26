# Flux2Klein 9B Full Fine-tuning 配置说明

## 训练状态 (2026-01-21)

### ✅ 成功配置

**DeepSpeed ZeRO-3** 是当前工作的分布式训练方案:

```bash
bash run_fulltune_deepspeed.sh
```

**Loss 范围正常:**
- 初始: ~1.5-2.0
- 稳定后: ~0.4-0.9
- 这与 LoRA 训练参考值一致 ("从2.x降到1.x，之后一直在1.x到0.5左右波动")

### GPU 资源
- 4x A100 80GB
- 每个 GPU 使用 ~71-72GB
- 100% GPU 利用率
- 训练速度: ~7.3s/step

---

## 文件结构

### 启动脚本
- `run_fulltune_deepspeed.sh` - DeepSpeed ZeRO-3 (推荐)
- `run_fulltune_fsdp.sh` - FSDP (备用)

### Accelerate 配置
- `accelerate_cfg/deepspeed_zero3_bf16.yaml` - DeepSpeed 配置
- `accelerate_cfg/fsdp_bf16.yaml` - FSDP 配置

### 训练脚本
- `train_OpenSciDraw_fulltune.py` - 主训练脚本

### 配置文件
- `configs/260121/flux2klein_fulltune_local_debug.py` - 训练配置

### 验证函数
- `OpenSciDraw/validation_funcs/Flux2Klein_fulltune_validation_func.py`

---

## 关键修改

### 1. Loss 计算修复
Flow Matching 的目标应该是 `noise - latent`，不是 `latent - noise`:
```python
# train_iteration_func
target = noise - packed_latents  # 正确
# target = packed_latents - noise  # 错误，会导致 Loss 过高 (~6.x)
```

### 2. DeepSpeed batch_sampler 兼容性
使用 `batch_sampler` 时需要手动设置 `train_micro_batch_size_per_gpu`:
```python
if is_deepspeed and hasattr(accelerator.state, 'deepspeed_plugin'):
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.train_batch_size
```

### 3. 分布式训练模式检测
```python
distributed_type_str = str(getattr(accelerator.state, 'distributed_type', 'NO'))
is_fsdp = 'FSDP' in distributed_type_str
is_deepspeed = 'DEEPSPEED' in distributed_type_str
```

### 4. 8-bit Adam 不兼容分布式训练
```python
use_8bit = config.use_8bit_adam and not is_fsdp and not is_deepspeed
```

---

## 验证功能

验证图像保存到:
```
output/flux2klein_fulltune_debug/validation_images/step_XXXXXX/
```

每 `validation_steps` (默认 100) 步运行一次验证。

---

## 注意事项

1. **FSDP + bf16 混合精度**: 有 gradient dtype mismatch 问题，需要使用 `mixed_precision: 'no'` 并手动使用 bf16
2. **DeepSpeed ZeRO-3**: 推荐使用，稳定性更好
3. **Validation 错误**: 推理时可能有 `'weight' must be 2-D` 错误，这是因为模型被 DeepSpeed 分片，不影响训练

---

## 快速启动

```bash
cd /home/v-yuxluo/WORK_local/ArXivQwenImage
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux2

# 使用 DeepSpeed (推荐)
bash run_fulltune_deepspeed.sh

# 或使用 FSDP (备用)
# bash run_fulltune_fsdp.sh
```
