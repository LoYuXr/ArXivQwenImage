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
class ArXiVParquetDatasetV2(Dataset):
    def __init__(self, base_dir, parquet_base_path, num_workers=16, 
                 num_train_examples=1000000, debug_mode=False, is_main_process=False, stat_data=False):
        self.base_path = Path(base_dir)
        self.data_base_path = self.base_path / parquet_base_path
        
        ## 首先读入所有的.parquet，现在不大，不需要读snapshot
        print(f"🔍 Building metadata from all parquet files in {self.data_base_path}...")
        year_dirs = sorted([d for d in self.data_base_path.iterdir() if d.is_dir()])
        all_paths = []
        for y_dir in year_dirs:
            all_paths.extend(sorted(y_dir.glob("*.parquet")))
        if debug_mode: all_paths = all_paths[:200]
        
        print(f"⏳ Loading/parsing metadata (parquet: path only) from {len(all_paths)} parquet files...")
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
                        default_key=["caption", "cache_path", "latent_shape", "text_embeds_shape", "bucket_w", "bucket_h", "aspect_ratio"]
                        ):
        meta_list = []
        def load_one_file(path):
            try:
                pf = pq.ParquetFile(path)
                # 读取 schema 获取真正的 top-level 列名 (pf.schema.names 会返回嵌套字段名如 'element')
                available_columns = [field.name for field in pf.schema_arrow]
                
                # 如果 text_embeds_shape 不存在但 prompt_embeds_shape 存在，做替换
                columns_to_read = default_key.copy()
                rename_map = {}
                
                if 'text_embeds_shape' not in available_columns and 'prompt_embeds_shape' in available_columns:
                    columns_to_read = [c if c != 'text_embeds_shape' else 'prompt_embeds_shape' for c in columns_to_read]
                    rename_map['prompt_embeds_shape'] = 'text_embeds_shape'
                
                df = pf.read(columns=columns_to_read).to_pandas()
                
                # 重命名列，统一为 text_embeds_shape
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
        # 统计每个桶的样本数
        counts = self.meta_df.groupby(['bucket_h', 'bucket_w']).indices
        valid_indices = []
        
        for bucket_key, indices in counts.items():
            # 计算在分布式环境下，这个桶能凑出多少个完整的 Batch
            # 每个桶至少需要：batch_size * num_replicas * 2 个样本才能保证每张卡都能分到
            total_needed = batch_size * num_replicas * 2
            if len(indices) >= total_needed:
                # 只有样本数足够的桶才保留
                valid_indices.extend(indices)
        
        # 更新 meta_df，只保留有效样本
        self.meta_df = self.meta_df.iloc[valid_indices].reset_index(drop=True)
        print(f"Filtered dataset: {len(self.meta_df)} samples remaining.")
        
        #

    def _read_sample_npz(self, npz_path):
        #tbd
        cache_latent_path = self.data_base_path / npz_path
        try:
            with  np.load(cache_latent_path, allow_pickle=True) as npz_data:
                
                latents_np = npz_data['latents'].astype(np.float32)
                
                # Support both 'text_embeds' (QwenImage) and 'prompt_embeds' (Flux2Klein)
                if 'text_embeds' in npz_data:
                    text_embeds_np = npz_data['text_embeds'].astype(np.float16)
                elif 'prompt_embeds' in npz_data:
                    text_embeds_np = npz_data['prompt_embeds'].astype(np.float16)
                else:
                    raise KeyError("Neither 'text_embeds' nor 'prompt_embeds' found in npz file")
                
                # Support both text_mask (QwenImage) and text_ids (Flux2Klein)
                # Priority: text_mask > text_ids > create default
                if 'text_mask' in npz_data:
                    text_mask = npz_data['text_mask'].astype(np.int8)
                    text_ids = None
                elif 'text_ids' in npz_data:
                    # Flux2Klein uses text_ids for RoPE position encoding
                    text_ids = npz_data['text_ids'].astype(np.float16)
                    # Create text_mask as all ones (all tokens valid)
                    seq_len = text_embeds_np.shape[0]
                    text_mask = np.ones((seq_len,), dtype=np.int8)
                else:
                    # Fallback: create default mask
                    seq_len = text_embeds_np.shape[0]
                    text_mask = np.ones((seq_len,), dtype=np.int8)
                    text_ids = None
                
                return latents_np, text_embeds_np, text_mask, text_ids
        except Exception as e:
            print(f"❌ Error reading npz {npz_path}: {e}")
            return None, None, None, None
        
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
            print(f"❌ Failed to load sample at index {index}. Use a zero latent and embed as fallback.")
            latents = np.zeros(tuple(map(int, meta_row['latent_shape'])), dtype=np.float32)
            # 补丁：
            text_embeds = np.zeros(tuple(map(int, meta_row['text_embeds_shape'])), dtype=np.float16)
            text_mask = np.zeros((text_embeds.shape[0],), dtype=np.int8)
            text_ids = None
            
        
        latents = torch.from_numpy(latents).reshape(list(map(int, meta_row['latent_shape'])))
        expected = int(np.prod(meta_row['text_embeds_shape']))
        actual = text_embeds.size

        if actual != expected:
            # print(
            #     f"[BAD SAMPLE]\n"
            #     f"cache_path={meta_row['cache_path']}\n"
            #     f"text_embeds.size={actual}\n"
            #     f"text_embeds_shape(meta)={meta_row['text_embeds_shape']}"
            # )
            text_embeds = torch.from_numpy(text_embeds)
            L_embed = text_embeds.shape[0]
            L_mask = text_mask.shape[0]
            L = min(L_embed, L_mask)

            if L_embed != L_mask:
                print(
                    f"[TRIM TEXT]\n"
                    f"cache_path={meta_row['cache_path']}\n"
                    f"L_embed={L_embed}, L_mask={L_mask} -> use L={L}"
                )
                
                text_embeds = text_embeds[:L]
                text_mask = text_mask[:L]   ###LYX HINT 先打上补丁！！！ 现在还错着呢，数据的制作！
                if text_ids is not None:
                    text_ids = text_ids[:L]
        
        else:
    
            text_embeds = torch.from_numpy(text_embeds).reshape(list(map(int, meta_row['text_embeds_shape'])))
        text_mask = torch.from_numpy(text_mask)
        
        result = {
            "latents": latents,
            "text_embeds": text_embeds,
            "text_mask": text_mask,
            "bucket_size": (meta_row['bucket_h'], meta_row['bucket_w']),
            "aspect_ratio": meta_row['aspect_ratio'],
            "caption": meta_row['caption']
        }
        
        # Add text_ids for Flux2Klein (RoPE position encoding)
        if text_ids is not None:
            result["text_ids"] = torch.from_numpy(text_ids)
        
        return result

    def collate_fn(self, batch):
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

    
@DATASETS.register_module()
class DistributedBucketSamplerV2(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, drop_last=True, shuffle=True, seed=42):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas or (dist.get_world_size() if dist.is_initialized() else 1)
        self.rank = rank if rank is not None else (dist.get_rank() if dist.is_initialized() else 0)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed  # Base seed for reproducibility
        self.epoch = 0

        self.groups = self.dataset.meta_df.groupby(['bucket_h', 'bucket_w']).indices

    def __iter__(self):
        # --- 3. 分布式同步核心：所有进程必须共享同一个 RNG ---
        # 使用 seed + epoch 来确保可复现性
        combined_seed = self.seed + self.epoch
        g = torch.Generator()
        g.manual_seed(combined_seed)
        rng = random.Random(combined_seed)  # 使用 Python 的 random 保证 shuffling 一致

        all_batch_lists = []
        
        # 排序 keys 保证所有卡遍历桶的顺序绝对一致
        sorted_bucket_keys = sorted(self.groups.keys())

        for bucket_key in sorted_bucket_keys:
            indices = self.groups[bucket_key].tolist()
            
            if self.shuffle:
                rng.shuffle(indices) # 全局统一打乱桶内样本
            
            # 补齐逻辑：让每个桶都能被 world_size * batch_size 整除（针对 drop_last=True）
            # 这一步是为了保证各卡看到的 Batch 数量完全相等
            if self.drop_last:
                total_per_bucket = (len(indices) // (self.num_replicas * self.batch_size)) * (self.num_replicas * self.batch_size)
                indices = indices[:total_per_bucket]
            else:
                total_per_bucket = int(math.ceil(len(indices) / (self.num_replicas * self.batch_size))) * (self.num_replicas * self.batch_size)
                # 循环补齐
                indices += indices[:(total_per_bucket - len(indices))]

            # 分发到当前 Rank (例如 4卡，Rank 0 拿 0, 4, 8...)
            # 但注意：我们要先组成所有的 batch，再分配，防止跨桶
            bucket_batches = []
            for i in range(0, len(indices), self.batch_size * self.num_replicas):
                # 这一块包含了所有卡在当前位置的 batch
                chunk = indices[i : i + self.batch_size * self.num_replicas]
                # 当前卡取自己那一份
                my_batch = chunk[self.rank * self.batch_size : (self.rank + 1) * self.batch_size]
                if len(my_batch) == self.batch_size:
                    bucket_batches.append(my_batch)
            
            all_batch_lists.extend(bucket_batches)

        # --- 4. 桶间打乱同步 ---
        # 必须所有卡对 batch 序列进行完全相同的打乱，否则会因为分辨率顺序不同导致死锁
        if self.shuffle:
            rng.shuffle(all_batch_lists)
            
        return iter(all_batch_lists)

    def __len__(self):
        # 此处的计算逻辑必须与 __iter__ 严丝合缝
        total_batches = 0
        for bucket_key in self.groups:
            indices = self.groups[bucket_key]
            num_samples_per_replica = len(indices) // self.num_replicas
            if self.drop_last:
                total_batches += num_samples_per_replica // self.batch_size
            else:
                total_batches += int(math.ceil(num_samples_per_replica / self.batch_size))
        return total_batches

    def set_epoch(self, epoch):
        self.epoch = epoch