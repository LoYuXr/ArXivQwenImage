import gc
import torch
import wandb
from PIL import Image

from accelerate.logging import get_logger
from diffusers import QwenImagePipeline  # 引入标准的 Pipeline

# 假设 VALIDATION_FUNCS 定义在 OpenSciDraw.registry
from OpenSciDraw.registry import VALIDATION_FUNCS

logger = get_logger(__name__)

@VALIDATION_FUNCS.register_module()
def QwenImage_validation_func(
    vae,
    transformer,
    text_encoder,
    accelerator,
    scheduler,
    tokenizer,
    args
):
    # 1. 初始化标准 Pipeline
    pipeline = QwenImagePipeline(
        vae=vae,
        transformer=accelerator.unwrap_model(transformer),
        text_encoder=accelerator.unwrap_model(text_encoder),
        scheduler=scheduler,
        tokenizer=tokenizer,
    )

    logger.info(f"Running validation...")
    pipeline = pipeline.to(accelerator.device)
    
    # 确保 pipeline 使用正确的数据类型 (通常是 bfloat16 或 float16)
    if args.mixed_precision == "fp16":
        pipeline.to(dtype=torch.float16)
    elif args.mixed_precision == "bf16":
        pipeline.to(dtype=torch.bfloat16)

    image_logs = []

    # 2. 遍历验证提示词
    for validation_prompt_idx in range(len(args.validation_prompts)):
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None

        validation_prompt = args.validation_prompts[validation_prompt_idx]
        
        # 移除了 validation_box 的逻辑，因为标准 pipeline 通常不需要这个
        # 如果你需要简单的矩形生成，可能需要根据具体需求调整，这里按标准文生图处理

        with torch.no_grad():
            # 使用 autocast 确保推理精度与训练一致
            with torch.amp.autocast(accelerator.device.type, dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16):
                images = pipeline(
                    prompt=validation_prompt,
                    height=args.resolution_list[validation_prompt_idx][1],  # 使用 args 中的分辨率
                    width=args.resolution_list[validation_prompt_idx][0],
                    num_inference_steps=args.num_inference_steps,
                    true_cfg_scale=args.true_cfg_scale, 
                    negative_prompt=args.negative_prompt,
                    generator=generator,
                    output_type="pil", # 确保输出 PIL 图像
                    max_sequence_length=args.max_sequence_length
                ).images 

        image_logs.append(
            {
                "images": images,
                "caption": validation_prompt,
            }
        )

    # 3. WandB 日志记录
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            dict_to_log = {}
            for sample_idx, log in enumerate(image_logs):
                formatted_images = []
                images = log["images"]
                validation_prompt = log["caption"]
                
                for idx, image in enumerate(images):
                    try:
                        image = wandb.Image(image, caption=f"{validation_prompt} ({idx})")
                        formatted_images.append(image)
                    except Exception as e:
                        logger.warning(f"W&B image log failed: {e}")
                
                dict_to_log[f"validation_sample_{sample_idx}"] = formatted_images
            
            tracker.log(dict_to_log)

    # 4. 清理内存
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()