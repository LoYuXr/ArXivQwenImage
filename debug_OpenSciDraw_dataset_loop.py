# =========================================================
# Path & Environment
# =========================================================
import os
import sys
import os.path as osp
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(
    0,
    osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), "..")),
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# =========================================================
# Standard Library
# =========================================================
import copy
import json
import math
import shutil
import logging
import warnings
import itertools
import tempfile
from functools import partial
from contextlib import nullcontext

# =========================================================
# Third-party Libraries
# =========================================================
from tqdm.auto import tqdm

import torch
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
# Accelerate
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)

# Transformers / Diffusers
import transformers
import diffusers
from diffusers import (
    AutoencoderKLQwenImage,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
    offload_models,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import (
    load_or_create_model_card,
    populate_model_card,
)
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module

# PEFT / LoRA
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from peft.utils import get_peft_model_state_dict

# =========================================================
# Project: OpenSciDraw
# =========================================================
from OpenSciDraw.registry import (
    DATASETS,
    TRAIN_ITERATION_FUNCS,
    VALIDATION_FUNCS,
)
from OpenSciDraw.utils import (
    parse_config,
    unwrap_model,
    get_trainable_params,
    initialize_QwenImage_all_models,
    add_lora_and_load_ckpt_to_models,
)

# =========================================================
# MMEngine
# =========================================================
from mmengine import Config, DictAction

# =========================================================
# Logger & Optional WandB
# =========================================================
logger = get_logger(__name__)

if is_wandb_available():
    import wandb


def main(config):
    # config = Config.fromfile(config)
    # print('config:', config)

    if config.report_to == "wandb" and config.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )
    
    if torch.backends.mps.is_available() and config.mixed_precision == "bf16":
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(config.model_output_dir, config.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=config.model_output_dir,
        logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    
    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if config.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    
    #logger.info(accelerator.state, main_process_only=False)


    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.model_output_dir is not None:
            os.makedirs(config.model_output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    config.weight_dtype = weight_dtype

    if torch.backends.mps.is_available() and config.weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
        )
    
    # move models to the assigned device
    #logger.info(f"[INFO] move models to the assigned device")
    #to_kwargs = {"dtype": config.weight_dtype, "device": accelerator.device} if not config.offload else {"dtype": config.weight_dtype} # bf16
    
    ## 注意，如果使用的是parquet数据集，就不在gpu上load vae, text encoder, tokenizer, text encoding pipeline，当inference pipeline启动时再load
    if not config.use_parquet_dataset:
        logger.info(f"[INFO] Using normal dataset, move all models to device now")

    else:
        logger.info(f"[INFO] Using parquet dataset, delay moving some models to device until inference")
        
    

    dataset_args = config.dataset.copy()
    dataset_args['is_main_process'] = accelerator.is_main_process # 传入这个状态

    train_dataset = DATASETS.build(dataset_args)

    # train_dataset = DATASETS.build(config.dataset)
    
    # data sampler config refine
    sampler_configs = config.data_sampler
    # sampler_configs["num_replicas"] = accelerator.num_processes  # <--- 总卡数
    # sampler_configs["rank"] = accelerator.process_index      # <--- 当前卡是第几
    # config.data_sampler = sampler_configs
    # train_sampler = DATASETS.build(
    #     config.data_sampler, default_args={"dataset": train_dataset})
    sampler_configs["sampler"] = RandomSampler(train_dataset)
    sampler_configs["dataset"] = train_dataset
    config.data_sampler = sampler_configs

    train_sampler = DATASETS.build(config.data_sampler)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.dataloader_num_workers,
        collate_fn=train_dataset.collate_fn,
    )
    
    

    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    logger.info(f"  Train iteration function: {config.train_iteration_func_name}")
    logger.info(f"  Validation function : {config.validation_func_name}")
    global_step = 0
    first_epoch = 0
    
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    train_iteration_func = TRAIN_ITERATION_FUNCS.get(config.train_iteration_func_name)
    validation_func = VALIDATION_FUNCS.get(config.validation_func_name)

    seen_samples = 0
    success_samples = 0

    first_epoch = 0
    for epoch in range(first_epoch, config.num_train_epochs):
        print(f"\n======== Epoch {epoch} / {config.num_train_epochs} ========")

        if hasattr(train_dataloader.batch_sampler, 'set_epoch'):
            train_dataloader.batch_sampler.set_epoch(epoch)
        elif hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
            
        
        for step, batch in enumerate(train_dataloader):
            #print(f"\n--- Epoch {epoch} Step {step} ---  GPU {accelerator.device} bucket_size: {batch['bucket_size']}, latents shape: {batch['latents'].shape}, text input_ids shape: {batch['text_embeds'].shape}, text mask shape: {batch['text_mask'].shape}")
            batch_size = batch["latents"].shape[0]
            seen_samples += batch_size
            success_samples += batch_size
            
            if global_step >= config.max_train_steps:
                break
            
            global_step += 1
            progress_bar.update(1)
        

            accelerator.wait_for_everyone()


    # Save the lora layers
    accelerator.wait_for_everyone()
    seen_tensor = torch.tensor(seen_samples, device=accelerator.device)
    success_tensor = torch.tensor(success_samples, device=accelerator.device)

    accelerator.reduce(seen_tensor, reduction="sum")
    accelerator.reduce(success_tensor, reduction="sum")

    if accelerator.is_main_process:
        logger.info(
            f"[DATA STATS] Seen samples: {seen_tensor.item()} | "
            f"Success samples: {success_tensor.item()} | "
            f"Success rate: {success_tensor.item() / max(seen_tensor.item(), 1):.4f}"
        )
    
    accelerator.end_training()
    


if __name__ == "__main__":
    config = parse_config()
    main(config)
    