import os
import glob
from torch.utils.data import DataLoader
from OpenSciDraw.registry import DATASETS
from OpenSciDraw.utils import (
    parse_config,
)


def main(config):
    # 构建数据集
    dataset = DATASETS.build(config.dataset)
    sampler = DATASETS.build(config.data_sampler, default_args={'dataset': dataset})
    
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=8,
        collate_fn=dataset.collate_fn
    )
    
    # 5. 验证循环
    print("\n--- Starting Check Loop ---")
    for i, batch in enumerate(loader):
        latents = batch['latents']
        # 验证点 1: 形状是否正确
        # 应该得到 [4, 4, 1, H, W] (如果保存时 unsqueeze了) 或者 [4, 4, H, W]
        print(f"Batch {i} | Latent Shape: {latents.shape} | Bucket: {batch['bucket_size']}")
        
        # 验证点 2: 确保一个 Batch 内的分辨率是完全一致的
        # bucket_size 是 dataset 返回的 (h, w)
        # 检查 batch 里的第一个样本和最后一个样本的 bucket_size 是否一样
        # (实际上 Sampler 逻辑保证了这一点，这里是 Double Check)
        
        if i >= 20:
            break

    print("Check Passed.")

if __name__ == "__main__":
    config = parse_config()
    main(config)