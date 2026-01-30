# Test Scripts for Flux2Klein Inference & EMA

This directory contains scripts for inference and EMA (Exponential Moving Average) computation for Flux2Klein 9B transformer models.

## Files

### EMA Computation

- **`compute_ema_amlt.py`** - Compute EMA weights from multiple checkpoints
  - Processes checkpoints from **oldest to newest**
  - Supports auto-detection of checkpoints from folder
  - Formula: `ema = decay * ema + (1-decay) * current`
  - Can be run locally or via AMLT

### Inference Scripts

- **`test_flux2klein_fulltune_inference.py`** - Single checkpoint inference
- **`test_flux2klein_ema_inference.py`** - EMA model inference  
- **`test_flux2klein_local_4gpu.py`** - 4-GPU parallel inference for GA4 checkpoints
- **`test_flux2klein_local_4gpu_ga1.py`** - 4-GPU parallel inference for GA1 checkpoints
- **`test_flux2klein_local_4gpu_ga2.py`** - 4-GPU parallel inference for GA2 checkpoints

---

## EMA Computation

### Method

EMA blends multiple checkpoints to create a smoothed model:

```
ema = decay * ema + (1 - decay) * current_checkpoint
```

Checkpoints are processed from **oldest to newest** (ascending step order).

### Usage

**Local (auto-detect checkpoints):**
```bash
python test/compute_ema_amlt.py \
    --experiment_name 260124_fluxklein9B_fulltune_GA4 \
    --fixed_decay 0.9 \
    --output_dir /path/to/output/ema_ga4 \
    --local_checkpoint_base /path/to/experiments/260124_fluxklein9B_fulltune_GA4
```

**AMLT (see `amlt/ema_gpu_job.yaml`):**
```bash
amlt run amlt/ema_gpu_job.yaml :ema_ga4 :ema_ga2 :ema_ga1 ema_experiment -d visionflow_basic -y
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--experiment_name` | Name of experiment (for metadata) |
| `--checkpoint_steps` | Manual list of steps (optional, comma-separated) |
| `--fixed_decay` | Fixed decay between consecutive checkpoints (e.g., 0.9) |
| `--per_step_decay` | Per-step decay (e.g., 0.9999), used if `fixed_decay` not set |
| `--output_dir` | Output directory for EMA model |
| `--local_checkpoint_base` | Path to checkpoint folder (auto-detects checkpoints) |

### Example: decay=0.9

With checkpoints [23000, 25000, 27000, 28500] and `--fixed_decay 0.9`:

1. Start with ckpt-23000 as initial EMA
2. `ema = 0.9 * ema + 0.1 * ckpt-25000`
3. `ema = 0.9 * ema + 0.1 * ckpt-27000`  
4. `ema = 0.9 * ema + 0.1 * ckpt-28500`

The newer checkpoints have less weight (more decayed), while older checkpoints have more cumulative weight.

---

## Inference

### Single Checkpoint Inference

```bash
python test/test_flux2klein_fulltune_inference.py \
    --checkpoint_path /path/to/checkpoint-28500 \
    --output_dir /path/to/output \
    --prompts_file prompts.txt
```

### EMA Model Inference

```bash
python test/test_flux2klein_ema_inference.py \
    --ema_path /path/to/ema_model \
    --output_dir /path/to/output \
    --prompts_file prompts.txt
```

### 4-GPU Parallel Inference

Runs inference on 4 checkpoints in parallel using 4 GPUs:

```bash
python test/test_flux2klein_local_4gpu.py
```

---

## Directory Structure

After EMA computation, the output directory contains:

```
ema_output/
├── transformer/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors  (~17GB)
└── ema_metadata.json
```

The `ema_metadata.json` records:
- Experiment name
- Checkpoint steps used
- Decay value
- EMA method description
