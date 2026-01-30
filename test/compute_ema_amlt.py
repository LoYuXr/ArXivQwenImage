"""
EMA (Exponential Moving Average) Computation for Flux2Klein Transformer
Designed for AMLT CPU jobs - downloads checkpoints from Azure Blob and computes EMA.

Correct EMA implementation:
- Process checkpoints from OLDEST to NEWEST
- EMA formula: ema = decay^(step_diff) * ema + (1 - decay^(step_diff)) * current
- Where step_diff is the number of steps between checkpoints
- decay is per-step decay rate (e.g., 0.9999)

For example with checkpoints [23000, 25000, 27000, 28500] and decay=0.9999:
1. Start with checkpoint-23000 as initial EMA
2. step_diff = 25000-23000 = 2000, effective_decay = 0.9999^2000 ≈ 0.82
   ema = 0.82 * ema + 0.18 * ckpt_25000
3. step_diff = 27000-25000 = 2000, effective_decay = 0.9999^2000 ≈ 0.82
   ema = 0.82 * ema + 0.18 * ckpt_27000
4. step_diff = 28500-27000 = 1500, effective_decay = 0.9999^1500 ≈ 0.86
   ema = 0.86 * ema + 0.14 * ckpt_28500

Usage (local):
    python test/compute_ema_amlt.py \
        --experiment_name 260124_fluxklein9B_fulltune_GA4 \
        --checkpoint_steps "23000,25000,27000,28500" \
        --per_step_decay 0.9999 \
        --output_dir /path/to/output

Usage (AMLT):
    See amlt/ema_cpu_job.yaml
"""

import argparse
import os
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Compute EMA weights for transformer checkpoints")
    
    # Azure Blob settings
    parser.add_argument("--blob_account", type=str, default="mcgvisionflowsa",
                        help="Azure Blob storage account name")
    parser.add_argument("--blob_container", type=str, default="yuxuanluo",
                        help="Azure Blob container name")
    parser.add_argument("--sas_token", type=str, default=None,
                        help="SAS token for Azure Blob (can also use AZURE_SAS_TOKEN env var)")
    
    # Experiment settings
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Name of the experiment folder in blob (e.g., 260124_fluxklein9B_fulltune_GA4)")
    parser.add_argument("--checkpoint_steps", type=str, default=None,
                        help="Comma-separated list of checkpoint steps (oldest to newest). If not provided, auto-detect from folder.")
    
    # EMA settings
    parser.add_argument("--per_step_decay", type=float, default=0.9999,
                        help="Per-step EMA decay rate (default: 0.9999)")
    parser.add_argument("--fixed_decay", type=float, default=None,
                        help="Fixed decay rate between consecutive checkpoints (overrides per_step_decay)")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for EMA checkpoint")
    parser.add_argument("--upload_to_blob", action="store_true",
                        help="Upload result back to Azure Blob")
    
    # Local mode (skip download, use local checkpoints)
    parser.add_argument("--local_checkpoint_base", type=str, default=None,
                        help="Local path to checkpoints (skip Azure download)")
    
    return parser.parse_args()


def download_checkpoint(blob_url: str, local_path: str, sas_token: str) -> bool:
    """Download a checkpoint from Azure Blob using azcopy."""
    os.makedirs(local_path, exist_ok=True)
    
    full_url = f"{blob_url}?{sas_token}"
    
    cmd = ["azcopy", "copy", full_url, local_path, "--recursive"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  [ERROR] azcopy failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] azcopy timeout")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def fix_directory_structure(checkpoint_path: str):
    """Fix nested transformer/transformer directory structure from azcopy."""
    nested_path = os.path.join(checkpoint_path, "transformer", "transformer")
    target_path = os.path.join(checkpoint_path, "transformer")
    
    if os.path.exists(nested_path):
        print(f"  Fixing nested directory structure...")
        for item in os.listdir(nested_path):
            src = os.path.join(nested_path, item)
            dst = os.path.join(target_path, item)
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
        os.rmdir(nested_path)


def scan_checkpoints(base_path: str) -> list:
    """
    Scan a directory for checkpoint folders and return sorted step numbers.
    
    Looks for folders named 'checkpoint-XXXXX' and extracts step numbers.
    Returns list of step numbers sorted ascending (oldest to newest).
    """
    import re
    checkpoint_pattern = re.compile(r'^checkpoint-(\d+)$')
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path not found: {base_path}")
    
    steps = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            match = checkpoint_pattern.match(item)
            if match:
                step = int(match.group(1))
                # Verify it has transformer weights
                transformer_path = os.path.join(item_path, "transformer")
                if os.path.exists(transformer_path):
                    safetensors_files = [f for f in os.listdir(transformer_path) 
                                          if f.endswith(".safetensors")]
                    if safetensors_files:
                        steps.append(step)
    
    if not steps:
        raise ValueError(f"No valid checkpoint folders found in {base_path}")
    
    # Sort ascending (oldest to newest)
    steps = sorted(steps)
    print(f"  Found {len(steps)} checkpoints: {steps}")
    return steps


def load_safetensors_weights(checkpoint_path: str) -> OrderedDict:
    """Load all safetensors weights from a checkpoint directory."""
    transformer_path = os.path.join(checkpoint_path, "transformer")
    if not os.path.exists(transformer_path):
        transformer_path = checkpoint_path
    
    # Check for single file
    single_file = os.path.join(transformer_path, "diffusion_pytorch_model.safetensors")
    
    if os.path.exists(single_file):
        print(f"  Loading single safetensors file...")
        return load_file(single_file)
    
    # Look for sharded files
    shard_files = sorted([f for f in os.listdir(transformer_path) 
                          if f.startswith("diffusion_pytorch_model") and f.endswith(".safetensors")])
    
    if not shard_files:
        raise FileNotFoundError(f"No safetensors files found in {transformer_path}")
    
    print(f"  Loading {len(shard_files)} shard files...")
    weights = OrderedDict()
    for shard_file in shard_files:
        shard_path = os.path.join(transformer_path, shard_file)
        shard_weights = load_file(shard_path)
        weights.update(shard_weights)
    
    return weights


def save_safetensors_single(weights: OrderedDict, output_path: str):
    """Save weights as a single safetensors file."""
    os.makedirs(output_path, exist_ok=True)
    
    output_file = os.path.join(output_path, "diffusion_pytorch_model.safetensors")
    print(f"  Saving to {output_file}...")
    save_file(weights, output_file)
    
    # Get size
    size_gb = os.path.getsize(output_file) / (1024**3)
    print(f"  Saved! Size: {size_gb:.2f} GB")


def compute_ema_correct(
    checkpoint_paths: list,
    checkpoint_steps: list,
    per_step_decay: float,
    fixed_decay: float = None
) -> OrderedDict:
    """
    Compute EMA weights correctly from oldest to newest checkpoint.
    
    Args:
        checkpoint_paths: List of paths to checkpoints (sorted oldest to newest)
        checkpoint_steps: List of step numbers (sorted ascending)
        per_step_decay: Per-step EMA decay rate
        fixed_decay: Fixed decay rate between consecutive checkpoints (overrides per_step_decay)
    
    Returns:
        EMA-averaged weights
    """
    assert len(checkpoint_paths) == len(checkpoint_steps)
    assert checkpoint_steps == sorted(checkpoint_steps), "Steps must be in ascending order!"
    
    print(f"\n{'='*60}")
    print(f"Computing EMA (oldest to newest)")
    if fixed_decay is not None:
        print(f"Fixed decay: {fixed_decay} (between consecutive checkpoints)")
    else:
        print(f"Per-step decay: {per_step_decay}")
    print(f"Checkpoints: {checkpoint_steps}")
    print(f"{'='*60}")
    
    ema_weights = None
    prev_step = None
    
    for i, (ckpt_path, step) in enumerate(zip(checkpoint_paths, checkpoint_steps)):
        print(f"\n[{i+1}/{len(checkpoint_steps)}] Processing checkpoint-{step}")
        
        # Load current checkpoint weights
        current_weights = load_safetensors_weights(ckpt_path)
        
        if ema_weights is None:
            # First checkpoint (oldest) - use as initial EMA
            ema_weights = current_weights
            print(f"  Using as initial EMA weights ({len(ema_weights)} tensors)")
        else:
            # Calculate effective decay
            if fixed_decay is not None:
                # Use fixed decay between consecutive checkpoints
                effective_decay = fixed_decay
            else:
                # Calculate decay based on step difference
                step_diff = step - prev_step
                effective_decay = per_step_decay ** step_diff
                print(f"  Step diff: {step_diff}")
            
            blend_factor = 1 - effective_decay
            
            print(f"  Decay: {effective_decay:.4f}")
            print(f"  Blend factor (new weight): {blend_factor:.4f}")
            
            # Blend: ema = decay * ema + (1-decay) * current
            for key in tqdm(ema_weights.keys(), desc="  EMA blend"):
                if key in current_weights:
                    ema_weights[key] = effective_decay * ema_weights[key] + blend_factor * current_weights[key]
                else:
                    print(f"  [WARNING] Key {key} not found in checkpoint-{step}")
        
        prev_step = step
        
        # Free memory
        del current_weights
        import gc
        gc.collect()
    
    return ema_weights


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"EMA Transformer Computation (AMLT Job)")
    print(f"{'='*60}")
    print(f"  Experiment: {args.experiment_name}")
    print(f"  Fixed Decay: {args.fixed_decay}")
    print(f"  Per-step Decay: {args.per_step_decay}")
    print(f"  Output Dir: {args.output_dir}")
    
    # Determine checkpoint steps
    if args.checkpoint_steps:
        # Manual specification
        checkpoint_steps = [int(s.strip()) for s in args.checkpoint_steps.split(',')]
        checkpoint_steps = sorted(checkpoint_steps)
        print(f"  Checkpoint Steps (manual): {checkpoint_steps}")
    elif args.local_checkpoint_base:
        # Auto-detect from local folder
        print(f"\n  Auto-detecting checkpoints from: {args.local_checkpoint_base}")
        checkpoint_steps = scan_checkpoints(args.local_checkpoint_base)
        print(f"  Checkpoint Steps (auto): {checkpoint_steps}")
    else:
        print("[ERROR] Must provide --checkpoint_steps or --local_checkpoint_base for auto-detection")
        return
    
    print(f"{'='*60}")
    
    # Get SAS token
    sas_token = args.sas_token or os.environ.get("AZURE_SAS_TOKEN", "")
    
    # Determine checkpoint paths
    if args.local_checkpoint_base:
        # Local mode - use existing checkpoints
        print(f"\n[LOCAL MODE] Using checkpoints from: {args.local_checkpoint_base}")
        checkpoint_paths = [
            os.path.join(args.local_checkpoint_base, f"checkpoint-{step}")
            for step in checkpoint_steps
        ]
    else:
        # Azure mode - download checkpoints
        print(f"\n[AZURE MODE] Downloading checkpoints from blob...")
        
        if not sas_token:
            print("[ERROR] SAS token required for Azure mode!")
            print("Set via --sas_token or AZURE_SAS_TOKEN environment variable")
            return
        
        # Create temp directory for downloads
        temp_dir = tempfile.mkdtemp(prefix="ema_checkpoints_")
        print(f"  Temp directory: {temp_dir}")
        
        checkpoint_paths = []
        blob_base = f"https://{args.blob_account}.blob.core.windows.net/{args.blob_container}/experiments/{args.experiment_name}"
        
        for step in checkpoint_steps:
            print(f"\n  Downloading checkpoint-{step}...")
            blob_url = f"{blob_base}/checkpoint-{step}/transformer"
            local_path = os.path.join(temp_dir, f"checkpoint-{step}")
            
            if download_checkpoint(blob_url, local_path, sas_token):
                fix_directory_structure(local_path)
                checkpoint_paths.append(local_path)
            else:
                print(f"  [ERROR] Failed to download checkpoint-{step}")
                return
    
    # Verify all checkpoints exist
    for path, step in zip(checkpoint_paths, checkpoint_steps):
        transformer_path = os.path.join(path, "transformer")
        if not os.path.exists(transformer_path):
            transformer_path = path
        
        safetensors_file = os.path.join(transformer_path, "diffusion_pytorch_model.safetensors")
        shard_files = [f for f in os.listdir(transformer_path) 
                       if f.startswith("diffusion_pytorch_model") and f.endswith(".safetensors")]
        
        if not os.path.exists(safetensors_file) and not shard_files:
            print(f"[ERROR] No safetensors files found for checkpoint-{step}")
            return
        
        print(f"  ✓ checkpoint-{step} verified")
    
    # Compute EMA
    ema_weights = compute_ema_correct(
        checkpoint_paths=checkpoint_paths,
        checkpoint_steps=checkpoint_steps,
        per_step_decay=args.per_step_decay,
        fixed_decay=args.fixed_decay
    )
    
    if ema_weights is None:
        print("\n[ERROR] EMA computation failed!")
        return
    
    # Create output directory
    output_transformer_dir = os.path.join(args.output_dir, "transformer")
    os.makedirs(output_transformer_dir, exist_ok=True)
    
    # Copy config.json from newest checkpoint
    newest_ckpt_path = checkpoint_paths[-1]
    config_candidates = [
        os.path.join(newest_ckpt_path, "transformer", "config.json"),
        os.path.join(newest_ckpt_path, "config.json"),
    ]
    for config_src in config_candidates:
        if os.path.exists(config_src):
            config_dst = os.path.join(output_transformer_dir, "config.json")
            shutil.copy2(config_src, config_dst)
            print(f"\nCopied config.json from checkpoint-{checkpoint_steps[-1]}")
            break
    
    # Save EMA weights as single file
    print(f"\nSaving EMA weights...")
    save_safetensors_single(ema_weights, output_transformer_dir)
    
    # Create metadata file
    metadata = {
        "experiment_name": args.experiment_name,
        "checkpoint_steps": checkpoint_steps,
        "per_step_decay": args.per_step_decay,
        "ema_method": "oldest_to_newest",
        "description": "EMA computed from oldest to newest checkpoint, with decay based on step differences"
    }
    metadata_path = os.path.join(args.output_dir, "ema_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Upload to blob if requested
    if args.upload_to_blob and sas_token:
        print(f"\nUploading EMA result to blob...")
        # Implementation for uploading back to blob
        pass
    
    print(f"\n{'='*60}")
    print(f"EMA computation complete!")
    print(f"Output saved to: {args.output_dir}")
    print(f"{'='*60}")
    
    # Cleanup temp directory if in Azure mode
    if not args.local_checkpoint_base and 'temp_dir' in locals():
        print(f"\nCleaning up temp directory...")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
