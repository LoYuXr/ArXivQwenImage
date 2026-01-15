import os
import os.path as osp
import sys
# Add the project directory to the Python path to simplify imports without manually setting PYTHONPATH.
sys.path.insert(
    0, osp.abspath(
        osp.join(osp.dirname(osp.abspath(__file__)), "..")
    ),
)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import copy
import math
import shutil
import logging
from tqdm.auto import tqdm
from functools import partial

import torch
import torch.utils.checkpoint

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
import transformers
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    check_min_version,
)
from diffusers import FluxPipeline
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict


from OpenSciDraw.registry import (
    DATASETS,
    TRAIN_ITERATION_FUNCS,
    VALIDATION_FUNCS,
)
from OpenSciDraw.utils import (
    parse_config,
    unwrap_model,
    is_wandb_available,
    get_trainable_params,
    initialize_QwenImage_all_models,
    add_lora_and_load_ckpt_to_models,
)

from ray import train
from ray.train import Checkpoint
import tempfile
from ray import train as ray_train
from mmengine import Config, DictAction


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

    (
        vae,
        transformer,
        tokenizer,
        text_encoder,
        noise_scheduler,
        text_encoding_pipeline,
        vae_scale_factor,
    ) = initialize_QwenImage_all_models(config)
    
    latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)).to(accelerator.device) # 这个vae不大一样，16维的latent
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(accelerator.device)

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

                QwenImagePipeline.save_lora_weights(  ### 我们如何根据config.transformer_cfg = dict(type='QwenImageTransformer2DModel',)来动态获取QwenImagePipeline???
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
                transformer_ = QwenImageTransformer2DModel.from_pretrained( #同理，如何动态获取???
                    config.pretrained_model_name_or_path, subfolder="transformer"
                )
                transformer_.add_adapter(transformer_lora_config)

            lora_state_dict = QwenImagePipeline.lora_state_dict(input_dir) #同理，如何动态获取???

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

    train_dataset = DATASETS.build(config.dataset)
    
    # data sampler config refine
    sampler_configs = config.data_sampler
    sampler_configs["num_replicas"] = accelerator.num_processes  # <--- 总卡数
    sampler_configs["rank"] = accelerator.process_index      # <--- 当前卡是第几
    config.data_sampler = sampler_configs

    train_sampler = DATASETS.build(
        config.data_sampler, default_args={"dataset": train_dataset})
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.dataloader_num_workers,
        collate_fn=train_dataset.collate_fn,
    )
    
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )

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
        transformer.train()
        
        train_loss = 0.0

        current_batch_dataloader = copy.deepcopy(train_dataloader)
        # for step, batch in enumerate(train_dataloader):
        for step, batch in enumerate(current_batch_dataloader): # pixels (1, 3, 1, 1024, 1024); prompts ...

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

                    avg_loss = accelerator.gather(loss.repeat(bs)).mean()
                    train_loss += avg_loss.item() / config.gradient_accumulation_steps

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = transformer.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)

                    optimizer.step()

                    lr_scheduler.step()
                    optimizer.zero_grad()

            except torch.cuda.OutOfMemoryError:
                print("Skipping batch due to OOM...")
                torch.cuda.empty_cache()
                continue

            torch.cuda.empty_cache()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % config.checkpointing_steps == 0 or global_step == config.max_train_steps:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(config.model_output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(config.model_output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                if global_step % config.checkpointing_steps == 0 or global_step == config.max_train_steps:

                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        # checkpoint = None
                        # torch.save({"x": torch.randn(1024), "y": torch.randn(200)}, os.path.join(temp_checkpoint_dir, "checkpoint.ckpt"))
                        save_path = os.path.join(temp_checkpoint_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                        # checkpoint.save(args.output_dir)
                        if accelerator.is_main_process:
                            ray_train.report(
                                {'hello': 'world'},
                                checkpoint=checkpoint
                            )
                        else:
                            ray_train.report(
                                {'hello': 'world'},
                                checkpoint=None
                            )

                    # save_path = os.path.join(config.model_output_dir, f"checkpoint-{global_step}")
                    # accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                accelerator.wait_for_everyone()


            logs = {
                "loss": loss.detach().item(),
                "train_loss": train_loss, # from ART
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            train_loss = 0.0

            if global_step >= config.max_train_steps:
                break

            if accelerator.is_main_process:
                # print('start validation')
                if config.validation_prompts is not None and global_step % config.validation_steps == 0 or global_step == 1 or global_step == config.max_train_steps-1:
                    validation_func(
                        vae=vae,
                        transformer=transformer,
                        text_encoder=text_encoding_pipeline.text_encoder,
                        accelerator=accelerator,
                        scheduler = noise_scheduler,
                        tokenizer = text_encoding_pipeline.tokenizer,
                        args=config,
                    )
                # print('finish validation')
                # exit(0)
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

        QwenImagePipeline.save_lora_weights(
            save_directory=config.model_output_dir,
            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

    accelerator.end_training()


if __name__ == "__main__":
    config = parse_config()
    main(config)
    
    