"""
ArXiV Parquet Dataset V3 - For Flux2Klein with VAE h (save_h format)

This dataset handles parquet data that stores VAE encoder hidden states (h) 
instead of directly sampled latents. This allows re-sampling during training,
which avoids the issue of identical latents when batch_size > 1.

The NPZ files contain:
- vae_h: [2*C, H/8, W/8] - VAE encoder output (mean and logvar concatenated)
- prompt_embeds: [SeqLen, Hidden] - Text embeddings (dynamic length)
- text_ids: [SeqLen, 3] - Flux-specific position IDs
- attention_mask: [SeqLen] - Attention mask

During __getitem__, we reconstruct latents by:
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    posterior = DiagonalGaussianDistribution(h)
    latents = posterior.sample() * vae.config.scaling_factor
"""

import os
import pandas as pd
import numpy as np
import torch
import itertools
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow.parquet as pq
from OpenSciDraw.registry import DATASETS
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist
import math
import time
import random


@DATASETS.register_module()
class ArXiVParquetDatasetV3(Dataset):
    """
    Dataset for Flux2Klein parquet data with VAE h format (save_h).
    
    Key differences from V2:
    - Reads 'vae_h' from NPZ instead of 'latents'
    - Uses DiagonalGaussianDistribution to sample latents during training
    - Supports dynamic scaling_factor from VAE config
    """
    
    def __init__(
        self, 
        base_dir, 
        parquet_base_path, 
        vae_scaling_factor=0.3611,  # Flux2 VAE default scaling factor
        num_workers=16, 
        num_train_examples=1000000, 
        debug_mode=False, 
        is_main_process=False, 
        stat_data=False
    ):
        self.base_path = Path(base_dir)
        self.data_base_path = self.base_path / parquet_base_path
        self.vae_scaling_factor = vae_scaling_factor
        
        # Read only *_train.parquet files (verified/curated data)
        # This excludes *_test.parquet, *_rank*_part*.parquet (intermediate files)
        print(f"🔍 Building metadata from !!!!*_train.parquet!!!! files in {self.data_base_path}...")
        year_dirs = sorted([d for d in self.data_base_path.iterdir() if d.is_dir()])
        all_paths = []
        for y_dir in year_dirs:
            # Only load *_train.parquet files
            train_parquets = sorted(y_dir.glob("*_train.parquet"))
            all_paths.extend(train_parquets)
        if debug_mode: 
            all_paths = all_paths[:200]
        
        print(f"⏳ Loading/parsing metadata from {len(all_paths)} train parquet files...")
        df = self._parallel_load_parquet(all_paths, max_workers=num_workers, num_train_examples=num_train_examples)
        self.meta_df = df
        
        print(f"✅ Loaded {len(self.meta_df)} samples.")
        
        self._filter_small_buckets(batch_size=8, num_replicas=4)
        
        if stat_data and is_main_process:
            print(f"📊 Data Statistics:")
            bucket_counts = self.meta_df.groupby(['bucket_h', 'bucket_w']).size().reset_index(name='counts')
            print(bucket_counts)
            total_samples = len(self.meta_df)
            for _, row in bucket_counts.iterrows():
                h, w, count = row['bucket_h'], row['bucket_w'], row['counts']
                print(f" - Resolution {w}x{h}: {count} samples ({count/total_samples*100:.2f}%)")
                
    def __len__(self):
        return len(self.meta_df)
        
    def _parallel_load_parquet(
        self, 
        paths, 
        max_workers, 
        num_train_examples, 
        default_key=["caption", "cache_path", "vae_h_shape", "prompt_embeds_shape", "bucket_w", "bucket_h", "aspect_ratio"]
    ):
        """Load parquet files in parallel."""
        meta_list = []
        
        def load_one_file(path):
            try:
                pf = pq.ParquetFile(path)
                available_columns = [field.name for field in pf.schema_arrow]
                
                # Determine which columns to read
                columns_to_read = []
                for col in default_key:
                    if col in available_columns:
                        columns_to_read.append(col)
                    elif col == 'vae_h_shape' and 'latent_shape' in available_columns:
                        # Fallback for older format
                        columns_to_read.append('latent_shape')
                    elif col == 'prompt_embeds_shape' and 'text_embeds_shape' in available_columns:
                        columns_to_read.append('text_embeds_shape')
                
                df = pf.read(columns=columns_to_read).to_pandas()
                
                # Rename columns to standard names
                rename_map = {}
                if 'latent_shape' in df.columns and 'vae_h_shape' not in df.columns:
                    rename_map['latent_shape'] = 'vae_h_shape'
                if 'text_embeds_shape' in df.columns and 'prompt_embeds_shape' not in df.columns:
                    rename_map['text_embeds_shape'] = 'prompt_embeds_shape'
                if rename_map:
                    df = df.rename(columns=rename_map)
                
                df['source_file'] = str(path)
                df['local_index'] = range(len(df))
                return df
            except Exception as e:
                return f"Error: {path} | {str(e)}"
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(load_one_file, p): p for p in paths}
            for future in tqdm(as_completed(future_to_path), total=len(paths), desc="Scanning Parquet Files"):
                result = future.result()
                if isinstance(result, pd.DataFrame):
                    meta_list.append(result)
                    
        return pd.concat(meta_list, ignore_index=True).iloc[:num_train_examples]
    
    def _filter_small_buckets(self, batch_size, num_replicas):
        """Filter out buckets that don't have enough samples."""
        counts = self.meta_df.groupby(['bucket_h', 'bucket_w']).indices
        valid_indices = []
        
        for bucket_key, indices in counts.items():
            total_needed = batch_size * num_replicas * 2
            if len(indices) >= total_needed:
                valid_indices.extend(indices)
        
        self.meta_df = self.meta_df.iloc[valid_indices].reset_index(drop=True)
        print(f"Filtered dataset: {len(self.meta_df)} samples remaining.")

    def _read_sample_npz(self, npz_path):
        """
        Read NPZ file and sample latents from VAE h.
        
        Returns:
            latents: Sampled latents (shape: [C, H, W])
            text_embeds: Text embeddings (shape: [SeqLen, Hidden])
            text_mask: Attention mask (shape: [SeqLen])
            text_ids: Position IDs for RoPE (shape: [SeqLen, 3])
        """
        cache_latent_path = self.data_base_path / npz_path
        try:
            with np.load(cache_latent_path, allow_pickle=True) as npz_data:
                # Load VAE h and sample latents
                if 'vae_h' in npz_data:
                    vae_h = npz_data['vae_h'].astype(np.float32)
                    # Sample latents from DiagonalGaussianDistribution
                    latents_np = self._sample_latents_from_h(vae_h)
                elif 'latents' in npz_data:
                    # Fallback for older format
                    latents_np = npz_data['latents'].astype(np.float32)
                else:
                    raise KeyError("Neither 'vae_h' nor 'latents' found in npz file")
                
                # Load text embeddings
                if 'prompt_embeds' in npz_data:
                    text_embeds_np = npz_data['prompt_embeds'].astype(np.float16)
                elif 'text_embeds' in npz_data:
                    text_embeds_np = npz_data['text_embeds'].astype(np.float16)
                else:
                    raise KeyError("Neither 'prompt_embeds' nor 'text_embeds' found in npz file")
                
                # NOTE: Do NOT load text_ids from parquet!
                # The parquet format stores text_ids as [SeqLen, 3] but Flux2 transformer
                # expects [SeqLen, 4] format (T, H, W, L). Let the training code generate
                # the correct format dynamically using _prepare_text_ids().
                text_ids = None
                
                # Load attention mask
                if 'attention_mask' in npz_data:
                    text_mask = npz_data['attention_mask'].astype(np.int8)
                elif 'text_mask' in npz_data:
                    text_mask = npz_data['text_mask'].astype(np.int8)
                else:
                    # Create default mask (all ones)
                    seq_len = text_embeds_np.shape[0]
                    text_mask = np.ones((seq_len,), dtype=np.int8)
                
                return latents_np, text_embeds_np, text_mask, text_ids
                
        except Exception as e:
            print(f"❌ Error reading npz {npz_path}: {e}")
            return None, None, None, None
    
    def _sample_latents_from_h(self, vae_h: np.ndarray) -> np.ndarray:
        """
        Sample latents from VAE encoder hidden states using DiagonalGaussianDistribution.
        
        Args:
            vae_h: VAE encoder output [2*C, H, W] containing mean and logvar
            
        Returns:
            latents: Sampled latents [C, H, W]
            
        Note: Unlike standard diffusers VAE, Flux2 VAE does not use scaling_factor.
        The latents are returned unscaled, same as the original Flux2KleinPipeline.
        """
        # vae_h shape: [2*C, H, W] where first half is mean, second half is logvar
        # For Flux2 VAE: C=32, so vae_h is [64, H, W]
        h_tensor = torch.from_numpy(vae_h).unsqueeze(0)  # [1, 2*C, H, W]
        
        # Split into mean and logvar
        # Flux2 VAE uses channel-wise split
        mean, logvar = torch.chunk(h_tensor, 2, dim=1)  # Each: [1, C, H, W]
        
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        
        # Sample using reparameterization trick
        sample = mean + std * torch.randn_like(mean)
        
        # Note: Flux2 VAE does not use scaling_factor in encode/decode
        # The latents are already in the correct scale
        
        return sample.squeeze(0).numpy()  # [C, H, W]
        
    def get_data_info(self, index):
        index = index % len(self.meta_df)
        sample = self.meta_df.iloc[index]
        return sample

    def __getitem__(self, index):
        meta_row = self.get_data_info(index)
        latents, text_embeds, text_mask, text_ids = self._read_sample_npz(
            meta_row['cache_path']
        )
        
        if latents is None or text_embeds is None or text_mask is None:
            print(f"❌ Failed to load sample at index {index}. Use zero tensors as fallback.")
            # Create fallback tensors
            vae_h_shape = meta_row['vae_h_shape']
            # vae_h_shape is [2*C, H, W], latent shape is [C, H, W]
            latent_c = int(vae_h_shape[0]) // 2
            latent_h = int(vae_h_shape[1])
            latent_w = int(vae_h_shape[2])
            latents = np.zeros((latent_c, latent_h, latent_w), dtype=np.float32)
            
            text_shape = meta_row['prompt_embeds_shape']
            text_embeds = np.zeros(tuple(map(int, text_shape)), dtype=np.float16)
            text_mask = np.zeros((text_embeds.shape[0],), dtype=np.int8)
            text_ids = None
        
        # Convert to tensors
        latents = torch.from_numpy(latents)
        text_embeds = torch.from_numpy(text_embeds)
        text_mask = torch.from_numpy(text_mask)
        
        # Ensure shapes match
        L_embed = text_embeds.shape[0]
        L_mask = text_mask.shape[0]
        L = min(L_embed, L_mask)
        
        if L_embed != L_mask:
            text_embeds = text_embeds[:L]
            text_mask = text_mask[:L]
            if text_ids is not None:
                text_ids = text_ids[:L]
        
        result = {
            "latents": latents,
            "text_embeds": text_embeds,
            "text_mask": text_mask,
            "bucket_size": (meta_row['bucket_h'], meta_row['bucket_w']),
            "aspect_ratio": meta_row['aspect_ratio'],
            "caption": meta_row['caption']
        }
        
        # Add text_ids for Flux2Klein RoPE position encoding
        if text_ids is not None:
            result["text_ids"] = torch.from_numpy(text_ids)
        
        return result

    def collate_fn(self, batch):
        """Collate function with dynamic padding for text embeddings."""
        from torch.nn.utils.rnn import pad_sequence
        
        latents = torch.stack([x['latents'] for x in batch])
        embeds_list = [x['text_embeds'] for x in batch]
        masks_list = [x['text_mask'] for x in batch]
        
        padded_embeds = pad_sequence(embeds_list, batch_first=True, padding_value=0)
        padded_masks = pad_sequence(masks_list, batch_first=True, padding_value=0)

        result = {
            "latents": latents,
            "text_embeds": padded_embeds,
            "text_mask": padded_masks,
            "captions": [x['caption'] for x in batch],
            "bucket_size": batch[0]['bucket_size'],
            "aspect_ratio": batch[0]['aspect_ratio'],
        }
        
        # Add text_ids for Flux2Klein if present
        if 'text_ids' in batch[0] and batch[0]['text_ids'] is not None:
            text_ids_list = [x['text_ids'] for x in batch]
            padded_text_ids = pad_sequence(text_ids_list, batch_first=True, padding_value=0)
            result["text_ids"] = padded_text_ids
        
        return result
