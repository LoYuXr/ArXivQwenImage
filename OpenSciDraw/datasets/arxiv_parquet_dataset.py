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

'''
ParquetDataset:
我们打包数据集为多个 Parquet 文件，每个文件包含若干样本的 Latents 和 Text Embeds。
'''

@DATASETS.register_module()
class ArXiVParquetDataset(Dataset):
    def __init__(self, base_dir, parquet_base_path, num_workers=16, 
                 num_train_examples=1000000, debug_mode=False, is_main_process=False):
        self.base_path = Path(base_dir)
        self.data_base_path = self.base_path / parquet_base_path
        
        # --- 1. 缓存同步加固：防止多进程竞争写同一个文件 ---
        cache_file = self.data_base_path / "metadata_cache_v1.pkl"
        
        if not cache_file.exists():
            if is_main_process:
                print(f"🔍 [Main Process] Scanning parquet files in {self.data_base_path}...")
                year_dirs = sorted([d for d in self.data_base_path.iterdir() if d.is_dir()])
                all_paths = []
                for y_dir in year_dirs:
                    all_paths.extend(sorted(y_dir.glob("*.parquet")))
                if debug_mode: all_paths = all_paths[:200]
                
                df = self._parallel_load_metadata(all_paths, max_workers=num_workers, num_train_examples=num_train_examples)
                # 先写临时文件再重命名，防止其他进程读到残缺文件
                tmp_cache = str(cache_file) + ".tmp"
                df.to_pickle(tmp_cache)
                os.rename(tmp_cache, str(cache_file))
                self.meta_df = df
            else:
                print(f"⏳ [Rank {dist.get_rank() if dist.is_initialized() else 0}] Waiting for metadata cache...")
                while not cache_file.exists():
                    time.sleep(2)
                self.meta_df = pd.read_pickle(cache_file)
        else:
            self.meta_df = pd.read_pickle(cache_file)

        self.meta_df = self.meta_df.iloc[:num_train_examples]
        print(f"✅ Loaded {len(self.meta_df)} samples.")

    def _parallel_load_metadata(self, paths, max_workers, num_train_examples):
        meta_list = []
        def load_one_header(path):
            try:
                pf = pq.ParquetFile(path)
                df = pf.read(columns=['bucket_w', 'bucket_h', 'latent_shape', 'text_embeds_shape', 'image_path', 'caption']).to_pandas()
                df['source_file'] = str(path)
                df['local_index'] = range(len(df))
                return df
            except Exception as e:
                return f"Error: {path} | {str(e)}"

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(load_one_header, p): p for p in paths}
            for future in tqdm(as_completed(future_to_path), total=len(paths), desc="Scanning Headers"):
                result = future.result()
                if isinstance(result, pd.DataFrame):
                    meta_list.append(result)
        
        return pd.concat(meta_list, ignore_index=True).iloc[:num_train_examples]

    def _read_sample_with_retry(self, file_path, local_idx, columns, retries=5):
        # --- 2. I/O 强力加固：指数退避重试 ---
        for i in range(retries):
            try:
                with pq.ParquetFile(file_path) as pf:
                    # 必须 explicit 传入 columns，绝对不读无用列
                    return pf.read_row_group(0, columns=columns).slice(local_idx, 1).to_pydict()
            except Exception as e:
                if i < retries - 1:
                    sleep_time = random.uniform(0.5, 2.0) * (2 ** i) # 指数增加等待时间
                    time.sleep(sleep_time)
                    continue
                raise RuntimeError(f"❌ I/O 彻底失败: {file_path} after {retries} retries. Error: {e}")

    def __getitem__(self, index):
        meta_row = self.meta_df.iloc[index]
        pair = self._read_sample_with_retry(
            meta_row['source_file'], 
            meta_row['local_index'], 
            columns=['latents', 'text_embeds', 'text_mask']
        )
        
        latents = torch.from_numpy(np.frombuffer(pair['latents'][0], dtype=np.float32).copy()).reshape(list(map(int, meta_row['latent_shape'])))
        text_embeds = torch.from_numpy(np.frombuffer(pair['text_embeds'][0], dtype=np.float16).copy()).reshape(list(map(int, meta_row['text_embeds_shape'])))
        text_mask = torch.from_numpy(np.frombuffer(pair['text_mask'][0], dtype=np.int8).copy())
        
        return {
            "latents": latents, "text_embeds": text_embeds, "text_mask": text_mask,
            "bucket_size": (meta_row['bucket_h'], meta_row['bucket_w']),
            "caption": meta_row['caption']
        }

    def collate_fn(self, batch):
        from torch.nn.utils.rnn import pad_sequence
        latents = torch.stack([x['latents'] for x in batch])
        embeds_list = [x['text_embeds'] for x in batch]
        masks_list = [x['text_mask'] for x in batch]
        padded_embeds = pad_sequence(embeds_list, batch_first=True, padding_value=0)
        padded_masks = pad_sequence(masks_list, batch_first=True, padding_value=0)

        return {
            "latents": latents, "text_embeds": padded_embeds, "text_mask": padded_masks,
            "captions": [x['caption'] for x in batch],
            "bucket_size": batch[0]['bucket_size'],
        }
        
@DATASETS.register_module()
class BucketSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # --- 核心简化：直接利用 DataFrame 的列进行分组 ---
        print("Grouping by pre-computed bucket_w/h...")
        
        # 这一步极快，pandas 内部优化的 hash group
        # keys 会变成: (768, 1024), (512, 512) 等等
        self.groups = self.dataset.meta_df.groupby(['bucket_h', 'bucket_w']).indices
        
        print(f"Found {len(self.groups)} unique resolutions.")
        for k, v in list(self.groups.items())[:3]:
            print(f"  Bucket {k}: {len(v)} samples")

    def __iter__(self):
        batch_lists = []
        
        for bucket_key, indices in self.groups.items():
            # 转换成 list 以便操作
            indices = list(indices)
            
            # 1. 桶内 Shuffle (满足"同一分辨率下 shuffle")
            if self.shuffle:
                np.random.shuffle(indices)
            
            # 2. 生成 Batch
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                
                batch_lists.append(batch)
        
        # 3. 桶间 Shuffle (让训练数据在不同分辨率间随机切换)
        if self.shuffle:
            np.random.shuffle(batch_lists)
            
        for batch in batch_lists:
            yield batch

    def __len__(self):
        count = 0
        for indices in self.groups.values():
            if self.drop_last:
                count += len(indices) // self.batch_size
            else:
                count += (len(indices) + self.batch_size - 1) // self.batch_size
        return count
    
    
@DATASETS.register_module()
class DistributedBucketSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, drop_last=True, shuffle=True):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas or (dist.get_world_size() if dist.is_initialized() else 1)
        self.rank = rank if rank is not None else (dist.get_rank() if dist.is_initialized() else 0)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.epoch = 0

        # 分组逻辑必须在初始化时完成，且所有进程一致
        self.groups = self.dataset.meta_df.groupby(['bucket_h', 'bucket_w']).indices

    def __iter__(self):
        # --- 3. 分布式同步核心：所有进程必须共享同一个 RNG ---
        g = torch.Generator()
        g.manual_seed(self.epoch)
        rng = random.Random(self.epoch) # 使用 Python 的 random 保证 shuffling 一致

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