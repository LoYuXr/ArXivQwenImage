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

'''
ParquetDataset:
我们打包数据集为多个 Parquet 文件，每个文件包含若干样本的 Latents 和 Text Embeds。
'''

@DATASETS.register_module()
class ArXiVParquetDataset(Dataset):
    def __init__(
            self,
            base_dir: str,
            parquet_base_path: str,
            num_workers: int = 64,
            num_train_examples: int = 1000000,
            pin_memory: bool = True,
            persistent_workers: bool = True,
            debug_mode: bool = False
            ):
 
        self.base_path = Path(base_dir)
        self.data_base_path = self.base_path / parquet_base_path
        
        # 1. 快速获取文件列表 (Glob 还是比较快的)
        print(f"🔍 Scanning parquet files in {self.data_base_path}...")
        # 假设目录结构是 year/xxx.parquet
        parquet_paths = [sorted(year_base_path.glob("*.parquet")) for year_base_path in sorted(self.data_base_path.iterdir()) if year_base_path.is_dir()]
        self.parquet_paths = list(itertools.chain.from_iterable(parquet_paths))
        
        
        if debug_mode:
            self.parquet_paths = self.parquet_paths[:10]  # 仅用于测试，限制读取文件数
        
        if not self.parquet_paths:
            raise RuntimeError(f"No parquet files found in {self.data_base_path}")

        print(f"Found {len(self.parquet_paths)} files. Starting PARALLEL metadata loading (No cache)...")

        # 2. 并发扫描所有文件的 Header
        # 这一步是替代 Cache 文件的关键
        self.meta_df = self._parallel_load_metadata(self.parquet_paths, max_workers=num_workers, num_train_examples=num_train_examples)
        
        print(f"✅ Total valid samples loaded: {len(self.meta_df)}")
        
        # 3. 运行时文件句柄缓存 (LRU)
        # 避免 __getitem__ 时反复打开关闭文件
        self.parquet_handles = {} 

    def _parallel_load_metadata(self, paths, max_workers, num_train_examples):
        """
        利用多线程并发读取 Metadata
        """
        meta_list = []
        
        # 定义单个文件的读取函数
        def load_one_header(path):
            try:
                # 使用 PyArrow ParquetFile 只读元数据，极快
                pf = pq.ParquetFile(path)
                
                # 只读取必要的列用于分组，绝对不要读 latents
                # read_row_group(0) 或者 read() 配合 columns 参数
                # 注意：如果文件确实损坏，这里会抛出异常
                df = pf.read(columns=['bucket_w', 'bucket_h', 'latent_shape', 'text_embeds_shape', 'image_path', 'caption']).to_pandas()
                
                # 注入定位信息
                df['source_file'] = str(path)
                df['local_index'] = range(len(df))
                return df
            except Exception as e:
                # 仅打印简短错误，防止刷屏
                return f"Error: {path} | {str(e)}"

        # 启动线程池
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_path = {executor.submit(load_one_header, p): p for p in paths}
            
            # 使用 tqdm 显示进度条
            for future in tqdm(as_completed(future_to_path), total=len(paths), desc="Scanning Headers"):
                result = future.result()
                if isinstance(result, pd.DataFrame):
                    meta_list.append(result)
                else:
                    # 如果是错误字符串
                    print(f"⚠️ {result}")

        if not meta_list:
            raise RuntimeError("All files failed to load!")

        # 合并所有结果
        return pd.concat(meta_list, ignore_index=True).iloc[:num_train_examples]

    def _get_parquet_handle(self, file_path):
        """
        运行时缓存打开的文件句柄，防止频繁 Open/Close
        """
        if file_path in self.parquet_handles:
            return self.parquet_handles[file_path]
        
        # 简单的 LRU: 如果超过 16 个打开的文件，清理掉最早的一个
        if len(self.parquet_handles) > 16:
            key_to_remove = next(iter(self.parquet_handles))
            self.parquet_handles.pop(key_to_remove)
            
        pf = pq.ParquetFile(file_path) 
        self.parquet_handles[file_path] = pf
        return pf

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, index):
        # 1. Look up metadata
        meta_row = self.meta_df.iloc[index]
        file_path = meta_row['source_file']
        local_idx = meta_row['local_index']
        
        # 2. Get file handle
        pf = self._get_parquet_handle(file_path)
        
        # 3. Read data
        full_df = pf.read().to_pandas() 
        row = full_df.iloc[local_idx]
        
        # --- Restore Tensors (Corrected Version) ---
        
        # 1. Latents: Must be float32 because that's how parquet stores it
        latents_np = np.frombuffer(row['latents'], dtype=np.float32)
        # Force convert shape to list of ints
        latents_shape = [int(x) for x in meta_row['latent_shape']]
        latents = torch.from_numpy(latents_np.copy()).reshape(latents_shape)
        
        # 2. Text Embeds: Usually float16
        text_embeds_np = np.frombuffer(row['text_embeds'], dtype=np.float16)
        text_shape = [int(x) for x in meta_row['text_embeds_shape']]
        text_embeds = torch.from_numpy(text_embeds_np.copy()).reshape(text_shape)
        
        # 3. Text Mask: int8
        text_mask_np = np.frombuffer(row['text_mask'], dtype=np.int8)
        text_mask = torch.from_numpy(text_mask_np.copy())
        
        return {
            "latents": latents,
            "text_embeds": text_embeds,
            "text_mask": text_mask,
            "bucket_size": (meta_row['bucket_h'], meta_row['bucket_w']),
            "caption": meta_row['caption']
        }

    def collate_fn(self, batch):
        # 1. 处理 Latents (BucketSampler 保证了宽高一致，直接 Stack)
        latents = torch.stack([x['latents'] for x in batch])

        # 2. 处理 Text Embeds (变长序列 -> 动态 Padding)
        # 你的 Parquet 里存的是不同长度的序列，比如 [525, 3584] 和 [616, 3584]
        
        embeds_list = [x['text_embeds'] for x in batch]
        masks_list = [x['text_mask'] for x in batch]

        # 引入 pad_sequence 工具
        from torch.nn.utils.rnn import pad_sequence
        
        # batch_first=True: 输出 [Batch, Max_Len, Dim]
        # padding_value=0: 缺的地方补0 (Qwen 的 Mask 逻辑通常 1是有效，0是无效)
        padded_embeds = pad_sequence(embeds_list, batch_first=True, padding_value=0)
        padded_masks = pad_sequence(masks_list, batch_first=True, padding_value=0)   ####ASKASKASKASK!!! LYXASK

        # 此时 padded_embeds 的形状会自动变成 [Batch, max_len_in_this_batch, 3584]
        # 比如 [4, 616, 3584]，而不是固定的 2048，这能加速训练！

        return {
            "latents": latents,
            "text_embeds": padded_embeds,
            "text_mask": padded_masks,
            "captions": [x['caption'] for x in batch],
            "bucket_size": batch[0]['bucket_size'],  # 全部一样
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
    def __init__(
        self, 
        dataset, 
        batch_size, 
        num_replicas=None, 
        rank=None, 
        drop_last=False, 
        shuffle=True
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # 1. 获取分布式信息
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            try:
                # 尝试自动获取
                num_replicas = dist.get_world_size()
                rank = dist.get_rank()
            except:
                # 如果没初始化 DDP，默认为单卡模式
                print("Warning: Distributed not initialized, assuming single GPU.")
                num_replicas = 1
                rank = 0
                
        self.num_replicas = num_replicas
        self.rank = rank
        
        # 2. 预分组 (和之前一样，利用 pandas group)
        print(f"[Rank {self.rank}] Grouping by pre-computed bucket_w/h...")
        # self.groups 结构: { (h, w): [idx1, idx2, ...], ... }
        self.groups = self.dataset.meta_df.groupby(['bucket_h', 'bucket_w']).indices
        
        # 3. 计算总样本数 (为了 __len__)
        # 在 DDP 中，必须保证每张卡看到的 batch 数量是一样的，否则训练会卡死
        self.num_samples_per_replica = 0
        for indices in self.groups.values():
            # 每个桶里的数据先分给 N 张卡
            count_per_bucket = int(math.ceil(len(indices) / self.num_replicas))
            # 再看能组成多少个 batch
            if self.drop_last:
                self.num_samples_per_replica += (count_per_bucket // self.batch_size) * self.batch_size
            else:
                self.num_samples_per_replica += int(math.ceil(count_per_bucket / self.batch_size)) * self.batch_size
        
        self.total_size = self.num_samples_per_replica * self.num_replicas
        print(f"[Rank {self.rank}] Initialized. World Size: {self.num_replicas}, Rank: {self.rank}")

    def __iter__(self):
        # 确定性种子：保证不同 epoch 的 shuffle 结果不同，但不同 GPU 上的 shuffle 逻辑一致
        g = torch.Generator()
        g.manual_seed(self.epoch if hasattr(self, 'epoch') else 0)
        
        batch_lists = []
        
        for bucket_key, indices in self.groups.items():
            indices = list(indices)
            
            # --- DDP 核心逻辑 1: 对索引进行 Shuffle ---
            # 必须在切分前 shuffle，并且所有 GPU 使用相同的种子，确保数据被打散但视角一致
            if self.shuffle:
                # 使用 numpy 或 torch 的 shuffle，这里为了简单用 numpy
                # 注意：这里不能用 self.rank 做种子，必须全局统一
                indices = np.array(indices)
                np.random.seed(self.epoch if hasattr(self, 'epoch') else 0) 
                np.random.shuffle(indices)
                indices = indices.tolist()
            
            # --- DDP 核心逻辑 2: Padding (补齐) ---
            # 确保每个桶的数据量能被 num_replicas 整除，防止某张卡最后没数据读
            total_size = int(math.ceil(len(indices) / self.num_replicas)) * self.num_replicas
            # 如果不够，循环补齐 (Round Robin)
            indices += indices[:(total_size - len(indices))]
            assert len(indices) == total_size
            
            # --- DDP 核心逻辑 3: 切分 (Subsampling) ---
            # 当前卡只处理属于自己的一部分: index, index+world_size, ...
            # 例如 4卡: Rank 0 取 [0, 4, 8...], Rank 1 取 [1, 5, 9...]
            indices = indices[self.rank:total_size:self.num_replicas]
            assert len(indices) == len(indices) # just valid check
            
            # --- DDP 核心逻辑 4: 桶内再 Shuffle (可选) ---
            # 这一步是为了让当前卡内部的数据更随机
            if self.shuffle:
                np.random.seed(self.epoch + self.rank if hasattr(self, 'epoch') else 0)
                np.random.shuffle(indices)
                
            # --- 生成 Batch ---
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                
                # 处理最后一个不完整的 batch
                if len(batch) < self.batch_size:
                    if self.drop_last:
                        continue
                    # 如果不 drop_last，可能需要补齐或者直接返回小 batch
                    # 为了 DDP 安全，通常建议 drop_last=True 或者自行补齐
                    # 这里简化处理：允许返回小 batch (注意 collate_fn 可能会遇到问题，建议 drop_last=True)
                
                batch_lists.append(batch)
                
        # --- 全局 Batch 间 Shuffle ---
        # 打乱不同分辨率 Batch 的顺序，让模型交替看到不同分辨率
        if self.shuffle:
            np.random.shuffle(batch_lists)
            
        for batch in batch_lists:
            yield batch

    def __len__(self):
        # 返回 batch 的数量
        return self.num_samples_per_replica // self.batch_size
    
    def set_epoch(self, epoch):
        self.epoch = epoch