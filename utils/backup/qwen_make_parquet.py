import os
import json
import argparse
import math
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.multiprocessing as mp
import glob
from torch.utils.data import Dataset, DataLoader, Sampler
from diffusers import AutoencoderKLQwenImage, QwenImagePipeline
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration
from mmengine import Config

# ================= 1. 优化后的 Dataset (IO 减负版) =================

class ArxivDataset(Dataset):
    def __init__(self, data_list, img_root):
        self.data_list = data_list
        self.img_root = img_root

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        rel_path = item['processed_info']['processed_path']
        
        # 路径构建
        full_path = os.path.join(self.img_root, rel_path)
        
        # 尝试读取
        try:
            img = cv2.imread(full_path)
            # 容错逻辑：尝试去年的路径
            if img is None:
                alt_rel = "/".join(rel_path.split("/")[1:])
                alt_path = os.path.join(self.img_root, alt_rel)
                img = cv2.imread(alt_path)
            
            if img is None:
                return None

            # 预处理
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 127.5 - 1.0
            img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
            
            return {
                "pixel_values": torch.from_numpy(img),
                "original_item": item,
            }
        except Exception as e:
            # 捕获所有图像损坏异常，防止卡死
            print(f"Error reading {rel_path}: {e}")
            return None

def custom_collate_fn(batch):
    # 过滤 None (读取失败的图)
    batch = [x for x in batch if x is not None]
    if len(batch) == 0: return None
    
    try:
        pixel_tensors = torch.stack([x["pixel_values"] for x in batch])
    except RuntimeError:
        # 极罕见情况：bucket分组由bug导致尺寸不一致
        print(f"⚠️ Shape mismatch in batch: {[x['pixel_values'].shape for x in batch]}")
        return None

    items = [x["original_item"] for x in batch]
    return pixel_tensors, items

# ================= 2. BatchSampler (保持不变) =================

class PredefinedBatchSampler(Sampler):
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

# ================= 3. 模型加载 (保持不变) =================

def load_models(config_path, device):
    print(f"[{device}] 🚀 Loading models...")
    config = Config.fromfile(config_path)
    
    vae = AutoencoderKLQwenImage.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="vae",
        revision=config.revision, cache_dir=config.cache_dir, token=config.huggingface_token,
    ).to(device=device, dtype=torch.float16)
    vae.requires_grad_(False)
    
    tokenizer = Qwen2Tokenizer.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="tokenizer",
        revision=config.revision, cache_dir=config.cache_dir, token=config.huggingface_token,
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=config.revision, cache_dir=config.cache_dir, token=config.huggingface_token,
        torch_dtype=torch.float16
    ).to(device=device)
    text_encoder.requires_grad_(False)
    
    pipeline = QwenImagePipeline.from_pretrained(
        config.pretrained_model_name_or_path, vae=None, transformer=None,
        tokenizer=tokenizer, text_encoder=text_encoder, scheduler=None,
    )
    pipeline.to(device)
    
    return vae, pipeline

# ================= 4. Worker 逻辑 (修复 num_workers) =================

def worker_func(rank, gpu_data_items, gpu_batch_indices, args):
    device = f"cuda:{rank}"
    
    # 1. 断点续调初始化
    year_str = str(args.year)
    output_dir = os.path.join(args.output_root, year_str)
    os.makedirs(output_dir, exist_ok=True)
    
    existing_files = glob.glob(os.path.join(output_dir, f"{year_str}_rank{rank}_part*.parquet"))
    processed_count = 0
    next_part_idx = 0
    
    if existing_files:
        print(f"[{device}] Found existing files, calculating resume position...")
        for f in existing_files:
            try:
                # 仅读取 Metadata 列，加速扫描
                df = pd.read_parquet(f, columns=['image_path'])
                processed_count += len(df)
                curr = int(f.split("_part")[-1].split(".")[0])
                next_part_idx = max(next_part_idx, curr + 1)
            except: pass
    
    # 计算需要跳过的 batch
    batches_to_skip = 0
    accumulated_items = 0
    for batch in gpu_batch_indices:
        if accumulated_items + len(batch) <= processed_count:
            accumulated_items += len(batch)
            batches_to_skip += 1
        else:
            break
            
    if batches_to_skip > 0:
        print(f"[{device}] Skipping {batches_to_skip} batches ({accumulated_items} items) from previous run.")
        gpu_batch_indices = gpu_batch_indices[batches_to_skip:]
    
    if not gpu_batch_indices:
        print(f"[{device}] All tasks completed.")
        return

    # 2. 构建 DataLoader (关键修改)
    dataset = ArxivDataset(gpu_data_items, args.img_root)
    sampler = PredefinedBatchSampler(gpu_batch_indices)
    
    # [优化] 计算合理的 worker 数量
    # 总核数 / GPU数，通常 8-12 之间比较好，绝不要用 64
    num_cpu_workers = min(16, os.cpu_count() // args.gpus)
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_cpu_workers, # 降低 worker 数，避免进程爆炸
        collate_fn=custom_collate_fn,
        pin_memory=True,
        prefetch_factor=4, # 增加预取，让 GPU 不等 IO
        persistent_workers=True # 保持 worker 存活，避免反复创建销毁
    )

    # 3. 加载模型
    vae, text_pipeline = load_models(args.config_path, device)

    # 4. 循环处理
    results = []
    
    # 只在 Rank 0 显示详细进度条，其他简单打印，避免 log 混乱
    # disable_tqdm = (rank != 0) 
    iterator = tqdm(dataloader, desc=f"GPU {rank}", position=rank)
    
    for batch_data in iterator:
        if batch_data is None: continue
        
        pixel_tensors, items = batch_data
        pixel_tensors = pixel_tensors.unsqueeze(2).to(device, dtype=torch.float16)
        
        with torch.no_grad():
            # VAE
            dist = vae.encode(pixel_tensors).latent_dist
            latents = dist.sample().cpu().float().numpy()
            
            # Text
            captions = [item.get('generated_fig_desc', "") for item in items]
            prompt_embeds, prompt_masks = text_pipeline.encode_prompt(
                prompt=captions, max_sequence_length=2048
            )
            text_embeds_np = prompt_embeds.cpu().float().numpy().astype(np.float16)
            text_mask_np = prompt_masks.cpu().numpy().astype(np.int8)

        # Pack
        for i, item in enumerate(items):
            res = {
                "image_path": item['processed_info']['processed_path'],
                "caption": captions[i],
                "latents": latents[i].tobytes(),
                "latent_shape": list(latents[i].shape),
                "text_embeds": text_embeds_np[i].tobytes(),
                "text_embeds_shape": list(text_embeds_np[i].shape),
                "text_mask": text_mask_np[i].tobytes(),
                "bucket_w": item['processed_info']['bucket_w'],
                "bucket_h": item['processed_info']['bucket_h'],
                "aspect_ratio": item['processed_info']['aspect_ratio']
            }
            results.append(res)

        # Save
        if len(results) >= args.save_nums:
            save_path = os.path.join(output_dir, f"{year_str}_rank{rank}_part{next_part_idx}.parquet")
            try:
                # 原子写入
                pd.DataFrame(results).to_parquet(save_path + ".tmp", compression='zstd')
                os.rename(save_path + ".tmp", save_path)
            except Exception as e:
                print(f"Error saving {save_path}: {e}")
                
            results = []
            next_part_idx += 1

    # Final Save
    if results:
        save_path = os.path.join(output_dir, f"{year_str}_rank{rank}_part{next_part_idx}.parquet")
        pd.DataFrame(results).to_parquet(save_path, compression='zstd')
        print(f"[{device}] Done. Saved final part.")

# ================= 5. 主程序 (Main) =================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--data_root", type=str, default="/home/v-yuxluo/yuxuanluo/ArXiV_filtered_stages/stage4_aspect_quantize_filter/768_pretrain")
    parser.add_argument("--img_root", type=str, default="/home/v-yuxluo/yuxuanluo/ArXiV_filtered_stages/stage4_aspect_quantize_filter/768_pretrain")
    parser.add_argument("--output_root", type=str, default="/home/v-yuxluo/data/ArXiV_parquet/768_pretrain_latents")
    parser.add_argument("--config_path", type=str, default="configs/train_qwen_image_config.py")
    parser.add_argument("--batch_size", type=int, default=4) # A100 80G 推荐 32-48，64可能OOM
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--save_nums", type=int, default=2048)
    args = parser.parse_args()

    # 1. 读取 JSON
    year_str = str(args.year)
    src_dir = os.path.join(args.data_root, year_str)
    json_path = os.path.join(src_dir, f"all_data_{year_str}.json")
    if not os.path.exists(json_path):
        jsons = [f for f in os.listdir(src_dir) if f.endswith(".json")]
        if not jsons: return
        json_path = os.path.join(src_dir, jsons[0])

    print(f"Reading {json_path}...")
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    # 2. 按 AR 分组
    print("Partitioning data by Aspect Ratio...")
    ar_groups = {}
    for item in raw_data:
        w = item['processed_info']['bucket_w']
        h = item['processed_info']['bucket_h']
        key = f"{w}x{h}"
        if key not in ar_groups: ar_groups[key] = []
        ar_groups[key].append(item)

    sorted_keys = sorted(ar_groups.keys(), key=lambda k: len(ar_groups[k]), reverse=True)

    # 3. 贪心算法分配 & 生成 Batch
    gpu_data_items = [[] for _ in range(args.gpus)] 
    gpu_batch_indices = [[] for _ in range(args.gpus)] 
    gpu_image_counts = [0] * args.gpus
    
    print("Distributing tasks...")
    for key in sorted_keys:
        items = ar_groups[key]
        count = len(items)
        min_gpu_idx = gpu_image_counts.index(min(gpu_image_counts))
        
        start_offset = len(gpu_data_items[min_gpu_idx])
        gpu_data_items[min_gpu_idx].extend(items)
        gpu_image_counts[min_gpu_idx] += count
        
        # 生成 Batch (不跨 AR 组)
        for i in range(0, count, args.batch_size):
            real_bs = min(args.batch_size, count - i)
            indices = list(range(start_offset + i, start_offset + i + real_bs))
            gpu_batch_indices[min_gpu_idx].append(indices)

    # 4. 启动多进程
    mp.set_start_method('spawn', force=True)
    processes = []
    
    print(f"Starting {args.gpus} workers...")
    for rank in range(args.gpus):
        p = mp.Process(
            target=worker_func, 
            args=(rank, gpu_data_items[rank], gpu_batch_indices[rank], args)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    print("All Finished.")

if __name__ == "__main__":
    main()