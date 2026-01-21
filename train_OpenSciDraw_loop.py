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
from torch.utils.data import DistributedSampler
# Accelerate
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from accelerate import Accelerator, DataLoaderConfiguration

# Transformers / Diffusers
import transformers
import diffusers
from diffusers import BitsAndBytesConfig
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
    initialize_QwenImage_all_models,  # Legacy, kept for backward compatibility
    add_lora_and_load_ckpt_to_models,
    # Model Factory: Dynamic model loading
    ModelFactory,
    initialize_models,
    get_pipeline_class,
    get_transformer_class,
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

def log_system_status(accelerator, transformer, vae, text_encoder, config=None):
    """
    打印整齐的系统状态日志：区分 Total、Trainable 以及纯 LoRA 参数
    """
    if not accelerator.is_main_process:
        return

    def get_detailed_params(model, name_filter="lora"):
        if model is None: 
            return 0, 0, 0
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # 专门统计包含特定关键字的可训练参数
        specific_trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and name_filter in n)
        return total, trainable, specific_trainable

    def get_device(model):
        if model is None: return "N/A"
        try:
            return str(next(model.parameters()).device)
        except (StopIteration, RuntimeError):
            return "Mixed/Offloaded"

    # 1. 核心计算逻辑：在 Transformer 中识别 LoRA
    tr_total, tr_train, tr_lora = get_detailed_params(transformer, "lora")
    vae_total, vae_train, _ = get_detailed_params(vae)
    te_total, te_train, _ = get_detailed_params(text_encoder)

    # 2. 打印表格
    logger.info("\n" + "="*95)
    logger.info(f"{'Model Component':<25} | {'Device':<15} | {'Total (M)':<12} | {'Trainable (M)':<15}")
    logger.info("-" * 95)
    
    # 打印 Transformer 行，特别标注 LoRA 占比
    tr_info = f"{tr_total/1e6:>10.2f}M"
    tr_train_info = f"{tr_train/1e6:>13.4f}M"
    logger.info(f"{'Transformer (DiT)':<25} | {get_device(transformer):<15} | {tr_info} | {tr_train_info}")
    
    # 如果开启了 LoRA，打印一行补充信息
    if tr_lora > 0:
        lora_ratio = (tr_lora / tr_total) * 100
        logger.info(f"{'  └─ Pure LoRA weights':<25} | {' ': <15} | {' ': <12} | {tr_lora/1e6:>13.4f}M ({lora_ratio:.3f}%)")

    logger.info(f"{'VAE':<25} | {get_device(vae):<15} | {vae_total/1e6:>10.2f}M | {vae_train/1e6:>13.4f}M")
    
    if text_encoder:
        logger.info(f"{'Text Encoder':<25} | {get_device(text_encoder):<15} | {te_total/1e6:>10.2f}M | {te_train/1e6:>13.4f}M")
    else:
        logger.info(f"{'Text Encoder':<25} | {'Offloaded/None':<15} | {'-':>12} | {'-':>15}")

    logger.info("-" * 95)
    
    # 3. 验证与采样检查
    if tr_train > 0:
        logger.info("🔍 Checking Trainable Parameters (Sample):")
        sample_count = 0
        for name, param in transformer.named_parameters():
            if param.requires_grad:
                logger.info(f"   -> {name} (Shape: {list(param.shape)}, Dtype: {param.dtype})")
                sample_count += 1
                if sample_count >= 2: break
        
        if tr_lora == tr_train and tr_train > 0:
            logger.info("✅ LoRA layers detected and matched all trainable params.")
        elif tr_lora > 0 and tr_lora < tr_train:
            logger.warning(f"⚠️ Mixed Training: {tr_lora/1e6:.2f}M LoRA and {(tr_train-tr_lora)/1e6:.2f}M other params are trainable!")
    else:
        logger.error("❌ NO trainable parameters found! The model is fully frozen.")
    
    logger.info("="*95 + "\n")

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
    
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
        dataloader_config=dataloader_config,
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
    
    
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

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

    # =========================================================
    # Model Initialization via Model Factory
    # =========================================================
    # Use the unified model factory based on config.model_type
    # Supported types: 'QwenImage', 'Flux2Klein', etc.
    model_type = getattr(config, 'model_type', 'QwenImage')
    logger.info(f"[INFO] Using model type: {model_type}")
    
    model_factory = ModelFactory(config)
    (
        vae,
        transformer,
        tokenizer,
        text_encoder,
        noise_scheduler,
        text_encoding_pipeline,
        vae_scale_factor,
    ) = model_factory.load_all()
    
    # Get the Pipeline and Transformer classes for save/load hooks
    PipelineClass = model_factory.PipelineClass
    TransformerClass = model_factory.TransformerClass
    
    # Get latents mean/std (model-specific)
    latents_mean, latents_std = model_factory.get_latents_stats(vae, accelerator.device)

    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # test validity: link these to config for train_iteration_func
    config.vae_scale_factor = vae_scale_factor
    config.latents_mean = latents_mean
    config.latents_std = latents_std
    
    
    # move models to the assigned device
    logger.info(f"[INFO] move models to the assigned device")
    to_kwargs = {"dtype": config.weight_dtype, "device": accelerator.device} if not config.offload else {"dtype": config.weight_dtype} # bf16
    
    ## 注意，如果使用的是parquet数据集，就不在gpu上load vae, text encoder, tokenizer, text encoding pipeline，当inference pipeline启动时再load
    if not config.use_parquet_dataset:
        logger.info(f"[INFO] Using normal dataset, move all models to device now")
        vae.to(**to_kwargs)
        text_encoder.to(**to_kwargs)
    
    else:
        logger.info(f"[INFO] Using parquet dataset, delay moving some models to device until inference")
        
    transformer_to_kwargs = (
        {"device": accelerator.device}
        if config.bnb_quantization_config_path is not None
        else {"device": accelerator.device, "dtype": config.weight_dtype}
    )
    transformer.to(**transformer_to_kwargs)
    
    if config.use_lora:
        if config.lora_layers is not None:
            target_modules = [layer.strip() for layer in config.lora_layers.split(",")]
        else:
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

        # now we will add new LoRA weights the transformer layers
        transformer_lora_config = LoraConfig(
            r=config.rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        transformer.add_adapter(transformer_lora_config)
        
        
        def unwrap_model(model):
            model = accelerator.unwrap_model(model)
            model = model._orig_mod if is_compiled_module(model) else model
            return model

        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, model_output_dir):
            if accelerator.is_main_process:
                transformer_lora_layers_to_save = None
                modules_to_save = {}

                for model in models:
                    if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                        model = unwrap_model(model)
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                        modules_to_save["transformer"] = model
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                # Use dynamic PipelineClass from model factory
                PipelineClass.save_lora_weights(
                    model_output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                    **_collate_lora_metadata(modules_to_save),
                )

        def load_model_hook(models, input_dir):
            transformer_ = None

            if not accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()

                    if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                        model = unwrap_model(model)
                        transformer_ = model
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")
            else:
                # Use dynamic TransformerClass from model factory
                transformer_ = TransformerClass.from_pretrained(
                    config.pretrained_model_name_or_path, subfolder="transformer"
                )
                transformer_.add_adapter(transformer_lora_config)

            # Use dynamic PipelineClass for loading lora state dict
            lora_state_dict = PipelineClass.lora_state_dict(input_dir)

            transformer_state_dict = {
                f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
            }


            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

            # Make sure the trainable params are in float32. This is again needed since the base models
            # are in `weight_dtype`. More details:
            # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
            if config.mixed_precision == "fp16":
                models = [transformer_]
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(models)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        
    else:
        raise NotImplementedError('Only lora fine-tuning is supported right now.')
    
    # Make sure the trainable params are in float32.
    if config.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters())) # 列出了若干个lora weights


    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": config.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (config.optimizer.lower() == "prodigy" or config.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {config.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        config.optimizer = "adamw"

    if config.use_8bit_adam and not config.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {config.optimizer.lower()}"
        )

    if config.optimizer.lower() == "adamw":
        if config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )

    if config.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if config.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(config.adam_beta1, config.adam_beta2),
            beta3=config.prodigy_beta3,
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
            decouple=config.prodigy_decouple,
            use_bias_correction=config.prodigy_use_bias_correction,
            safeguard_warmup=config.prodigy_safeguard_warmup,
        )

    dataset_args = config.dataset.copy()
    dataset_args['is_main_process'] = accelerator.is_main_process # 传入这个状态

    train_dataset = DATASETS.build(dataset_args)

    # train_dataset = DATASETS.build(config.dataset)
    
    # data sampler config refine
    # sampler_configs = config.data_sampler
    # sampler_configs["num_replicas"] = accelerator.num_processes  # <--- 总卡数
    # sampler_configs["rank"] = accelerator.process_index      # <--- 当前卡是第几
    # config.data_sampler = sampler_configs
    # train_sampler = DATASETS.build(
    #     config.data_sampler, default_args={"dataset": train_dataset})
    base_sampler = DistributedSampler(
        train_dataset,
        num_replicas=1,#accelerator.num_processes,
        rank=0,#accelerator.process_index,
        shuffle=True,
        seed=config.seed
    )
    # sampler_configs = config.data_sampler.copy()
    # sampler_configs["sampler"] = base_sampler
    # sampler_configs["dataset"] = train_dataset
    # sampler_configs["num_replicas"] = accelerator.num_processes  # <--- 总卡数
    # sampler_configs["rank"] = accelerator.process_index      # <--- 当前卡是第几
    # config.data_sampler = sampler_configs
    mysampler_configs = {
        "type": "ArXiVMixScaleBatchSampler",
        "dataset": train_dataset,
        "batch_size": config.train_batch_size,
        "num_replicas": 1, #accelerator.num_processes,
        "rank": 0, #accelerator.process_index,
        "drop_last": True,
        "shuffle": True,
        "seed": config.seed
    }
    train_sampler = DATASETS.build(mysampler_configs)


    #train_sampler = DATASETS.build(config.data_sampler)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.dataloader_num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True # 建议开启提升效率
    )
    
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_sampler=train_sampler,
    #     num_workers=config.dataloader_num_workers,
    #     collate_fn=train_dataset.collate_fn,
    # )
    
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )
    
    ### log the model conditions
    log_system_status(accelerator, transformer, vae, text_encoder)

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    if accelerator.is_main_process:
        tracker_name = config.tracker_name
        accelerator.init_trackers(
            tracker_name, \
            config=vars(config), \
            init_kwargs={
            "wandb": {
                "name": getattr(config, "run_name", "experiment-001"),
                }
            } if config.report_to == "wandb" else None,
        )

    # Train!
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

    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(config.model_output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.model_output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            # first_epoch = global_step // num_update_steps_per_epoch

    else:
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


    first_epoch = 0
    for epoch in range(first_epoch, config.num_train_epochs):
        print(f"\n======== Epoch {epoch} / {config.num_train_epochs} ========")

        if hasattr(train_dataloader.batch_sampler, 'set_epoch'):
            train_dataloader.batch_sampler.set_epoch(epoch)
        elif hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
            
        transformer.train()
        
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]

            try:
                with accelerator.accumulate(models_to_accumulate):
                    loss, bs = train_iteration_func(
                        batch,
                        vae,
                        transformer,
                        text_encoding_pipeline,
                        noise_scheduler_copy,
                        accelerator,
                        config,
                    )
                    
                    # 1. 收集所有显卡的 loss 用于日志
                    # 注意：这里 loss 已经是标量了
                    avg_loss = accelerator.gather(loss.repeat(bs)).mean()
                    train_loss += avg_loss.item() / config.gradient_accumulation_steps

                    # 2. 反向传播
                    accelerator.backward(loss)
                    
                    # 3. 只有在梯度同步时（即达到累积步数时）才执行更新
                    if accelerator.sync_gradients:
                        params_to_clip = transformer.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                        
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                    
                    ################## debug only  ##################
                    # progress_bar.update(1)
                    # global_step += 1
                    # progress_bar.set_postfix(status="Scanning Data...")    
                    
                    # if step % 100 == 0:
                    # # 假设 batch['image_path'] 是一个 list
                    #     caption = batch['captions'][0] 
                    #     print(f"[Rank {accelerator.process_index}] Step {step}: {caption}")
                    
                    #################################################
                    
                    
                        # --- 进度条与检查点逻辑 ---
                        progress_bar.update(1)
                        global_step += 1
                        
                        # 只有在这里才记录并重置 train_loss
                        logs = {
                            "loss": avg_loss.item(), # 当前 step 的 loss
                            "train_loss": train_loss, # 累积后的平均 loss
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                        progress_bar.set_postfix(**logs)
                        accelerator.log(logs, step=global_step)
                        
                        train_loss = 0.0 # 重置位置：移到同步逻辑内部

                        # Checkpoint 保存逻辑 (保持你原有的不变)
                        if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                            if global_step % config.checkpointing_steps == 0:
                                # ... 保存代码 ...
                                save_path = os.path.join(config.model_output_dir, f"checkpoint-{global_step}")
                                accelerator.save_state(save_path)
                                
                                
                        # 如果画图

                        # 5. 验证逻辑 (仅限主进程)
                        if accelerator.is_main_process:
                            if config.validation_prompts is not None and global_step > 0 and \
                               (global_step % config.validation_steps == 0 or global_step == 1 or global_step == config.max_train_steps-1):
                                
                                # 临时移动模型 (如果需要)
                                if config.use_parquet_dataset:
                                    vae.to(accelerator.device)
                                    if hasattr(text_encoding_pipeline, 'text_encoder'):
                                        text_encoding_pipeline.text_encoder.to(accelerator.device)

                                validation_func(
                                    vae=vae,
                                    transformer=transformer,
                                    text_encoder=text_encoding_pipeline.text_encoder,
                                    accelerator=accelerator,
                                    scheduler = noise_scheduler,
                                    tokenizer = text_encoding_pipeline.tokenizer,
                                    args=config,
                                )

                                if config.use_parquet_dataset:
                                    vae.to("cpu")
                                    if hasattr(text_encoding_pipeline, 'text_encoder'):
                                        text_encoding_pipeline.text_encoder.to("cpu")
                                    torch.cuda.empty_cache()
                        

                    accelerator.wait_for_everyone() # 通常不需要每步都等，save_state 内部会处理

            except torch.cuda.OutOfMemoryError:
                print("Skipping batch due to OOM...")
                torch.cuda.empty_cache()
                optimizer.zero_grad() # OOM 后记得清空梯度
                continue

            if global_step >= config.max_train_steps:
                break
        

            accelerator.wait_for_everyone()


    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        modules_to_save = {}
        transformer = unwrap_model(transformer)
        if config.bnb_quantization_config_path is None:
            if config.upcast_before_saving:
                transformer.to(torch.float32)
            else:
                transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        modules_to_save["transformer"] = transformer

        # Use dynamic PipelineClass from model factory
        PipelineClass.save_lora_weights(
            save_directory=config.model_output_dir,
            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

    accelerator.end_training()

if __name__ == "__main__":
    config = parse_config()
    main(config)
    