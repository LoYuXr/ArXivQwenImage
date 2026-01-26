import os
import torch
import pandas as pd
import numpy as np
import glob
from diffusers import AutoencoderKLQwenImage
from mmengine import Config
from PIL import Image, ImageDraw
import torch.distributed as dist  # <--- 添加这一行

# ================= Configuration =================
# 1. 你的 Config 路径
CONFIG_PATH = "configs/train_qwen_image_config.py" 
# 2. Parquet 所在的文件夹 (请确认路径无误)
PARQUET_FOLDER = "/home/v-yuxluo/data/ArXiV_parquet/768_pretrain_latents/2015"
# 3. 结果保存文件名
OUTPUT_IMAGE = "check_result_final.png"
# =================================================

def load_vae(config_path, device):
    print(f"Loading VAE from {config_path}...")
    try:
        config = Config.fromfile(config_path)
    except Exception as e:
        print(f"Error loading config with mmengine: {e}")
        # 备用方案：如果 mmengine 加载失败，尝试直接硬编码 model name
        print("Trying to load directly from huggingface path if config fails...")
        # 这里的 path 需要你根据实际情况修改，如果 config 加载成功则忽略此块
        return None

    # 加载 VAE，强制使用 float16
    vae = AutoencoderKLQwenImage.from_pretrained(
        config.pretrained_model_name_or_path, 
        subfolder="vae",
        revision=config.revision, 
        cache_dir=config.cache_dir, 
        token=config.huggingface_token,
    ).to(device=device, dtype=torch.float16)
    vae.eval()
    return vae

def process_one_parquet(parquet_path, vae, device, num_samples=4):
    print(f"Reading {parquet_path}...")
    
    # 读取 parquet
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    
    vis_images = []
    samples = df.head(num_samples)
    
    for idx, row in samples.iterrows():
        try:
            # -------------------------------------------------
            # 1. 还原 Latents
            # -------------------------------------------------
            # [关键点 A]: Parquet 存的是 float32，必须用 float32 读取
            latents_np = np.frombuffer(row['latents'], dtype=np.float32)
            
            # [关键点 B]: Shape 必须转为 int list
            shape = [int(s) for s in row['latent_shape']]
            
            # 检查点：数据量是否对齐
            expected_size = np.prod(shape)
            if latents_np.size != expected_size:
                print(f"⚠️ Sample {idx} mismatch! Buffer: {latents_np.size}, Shape prod: {expected_size}")
                # 尝试通过截断或补零修复（仅用于调试），或者跳过
                continue

            # 转 Tensor -> Reshape -> 增加 Batch 维度
            latents = torch.from_numpy(latents_np.copy())
            latents = latents.reshape(shape).unsqueeze(0) # [1, C, T, H, W]
            
            # [关键点 C]: 强制转为 float16 并送入 GPU
            latents = latents.to(device=device, dtype=torch.float16)

            # -------------------------------------------------
            # 2. VAE Decode
            # -------------------------------------------------
            with torch.no_grad():
                # Qwen VAE Decode
                # 注意：如果生成时没有乘 scaling_factor，这里就不需要除。
                # 如果图片看起来灰蒙蒙或全黑，可以尝试解开下面这行的注释：
                # latents = latents / vae.config.scaling_factor 
                
                decoded = vae.decode(latents, return_dict=False)[0]
                
                # 取第一帧 [B, C, T, H, W] -> [B, C, H, W]
                img_tensor = decoded[:, :, 0, :, :]
            
            # -------------------------------------------------
            # 3. 转图片
            # -------------------------------------------------
            img_tensor = (img_tensor / 2 + 0.5).clamp(0, 1)
            img_tensor = img_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
            img_uint8 = (img_tensor[0] * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            
            # -------------------------------------------------
            # 4. 绘图验证
            # -------------------------------------------------
            caption = row['caption']
            emb_bytes = np.frombuffer(row['text_embeds'], dtype=np.float16) # Embeds 通常是 fp16
            emb_shape = row['text_embeds_shape']
            
            w, h = pil_img.size
            canvas = Image.new('RGB', (w, h + 80), (255, 255, 255))
            canvas.paste(pil_img, (0, 80))
            draw = ImageDraw.Draw(canvas)
            # plot latent shape
            draw.text((5, 5), f"Latent Shape: {shape}", fill=(0,0,0))
            # plot image resolution, embed shape, caption
            draw.text((5, 5), f"Res: {w}x{h}\nEmb: {emb_shape}\nCap: {caption}", fill=(0,0,0))
            
            vis_images.append(canvas)
            print(f"Sample {idx}: Success! {w}x{h}")
            
        except Exception as e:
            print(f"❌ Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()

    return vis_images

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 查找 Parquet
    files = glob.glob(os.path.join(PARQUET_FOLDER, "*.parquet"))
    if not files:
        print(f"No files found in {PARQUET_FOLDER}")
        return
    files.sort()
    target_file = files[0] # 取第一个文件
    
    # 加载模型
    vae = load_vae(CONFIG_PATH, device)
    
    # 执行处理
    images = process_one_parquet(target_file, vae, device, num_samples=4)
    
    # 保存结果
    if images:
        max_w = max(img.width for img in images)
        total_h = sum(img.height for img in images)
        final_grid = Image.new('RGB', (max_w, total_h), (255, 255, 255))
        
        y_offset = 0
        for img in images:
            final_grid.paste(img, (0, y_offset))
            y_offset += img.height
            
        final_grid.save(OUTPUT_IMAGE)
        print(f"\n✅ All done! Result saved to: {os.path.abspath(OUTPUT_IMAGE)}")
    else:
        print("\n❌ Failed to generate any images.")

if __name__ == "__main__":
    main()