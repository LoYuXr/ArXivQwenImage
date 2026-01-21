# OpenSciDraw Module Documentation

**OpenSciDraw** is the core framework module of ScienceFlow, providing a modular and extensible architecture for training diffusion models on scientific figure generation tasks.

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Module Structure](#module-structure)
- [Core Components](#core-components)
- [Adding New Models](#adding-new-models)
- [Adding New Datasets](#adding-new-datasets)
- [Custom Training Functions](#custom-training-functions)
- [Validation Functions](#validation-functions)
- [Utilities](#utilities)

## 🏗️ Architecture Overview

OpenSciDraw follows a **registry-based design pattern** where components are dynamically registered and instantiated based on configuration. This allows for:

- **Decoupling**: Each component is independent and can be developed separately
- **Extensibility**: New models, datasets, or training functions can be added without modifying core code
- **Flexibility**: Mix and match components via configuration files

```
┌─────────────────────────────────────────────────────────────┐
│                      Registry System                        │
│  (Centralized component registration and retrieval)        │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐     ┌──────────────┐
│   Datasets   │      │   Training   │     │  Validation  │
│              │      │   Functions  │     │  Functions   │
└──────────────┘      └──────────────┘     └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                      ┌──────────────┐
                      │  Model       │
                      │  Factory     │
                      └──────────────┘
```

## 📁 Module Structure

```
OpenSciDraw/
├── __init__.py                          # Module exports
├── registry.py                          # Component registry
│
├── datasets/                            # Dataset implementations
│   ├── __init__.py
│   ├── arxiv_parquet_dataset_v2.py     # Parquet-based dataset
│   ├── arxiv_mixed_scale_batch_sampler.py  # Custom sampler
│   └── base_dataset.py                 # Base dataset class
│
├── train_iteration_funcs/              # Model-specific training loops
│   ├── __init__.py
│   ├── Flux2Klein_train_iteration_func.py   # Flux2Klein training
│   ├── QwenImage_train_iteration_func.py    # QwenImage training (legacy)
│   └── base_train_iteration.py        # Base training interface
│
├── validation_funcs/                   # Validation and inference
│   ├── __init__.py
│   ├── generic_validation_func.py      # Generic validation
│   └── model_specific_validation.py    # Model-specific validation
│
└── utils/                              # Utilities and helpers
    ├── __init__.py
    ├── model_factory.py                # Dynamic model loading
    ├── model_utils.py                  # Model utilities
    ├── lora_utils.py                   # LoRA utilities
    ├── training_utils.py               # Training utilities
    └── general_util_funcs.py           # General utilities
```

## 🔧 Core Components

### 1. Registry System (`registry.py`)

The registry provides centralized component management:

```python
from OpenSciDraw.registry import DATASETS, TRAIN_ITERATION_FUNCS, VALIDATION_FUNCS

# Register a new component
@DATASETS.register_module()
class MyCustomDataset:
    def __init__(self, ...):
        pass

# Retrieve and instantiate a component
dataset = DATASETS.build(config.dataset)
train_func = TRAIN_ITERATION_FUNCS.get('MyTrainFunc')
```

**Available Registries:**
- `DATASETS`: Dataset classes and samplers
- `TRAIN_ITERATION_FUNCS`: Training iteration functions
- `VALIDATION_FUNCS`: Validation functions

### 2. Model Factory (`utils/model_factory.py`)

Dynamically loads and initializes models based on `model_type`:

```python
from OpenSciDraw.utils import ModelFactory

factory = ModelFactory(config)
vae, transformer, tokenizer, text_encoder, scheduler, text_pipeline, scale = factory.load_all()

# Model-specific latent statistics
latents_mean, latents_std = factory.get_latents_stats(vae, device)

# Get pipeline and transformer classes
PipelineClass = factory.PipelineClass
TransformerClass = factory.TransformerClass
```

**Supported Models:**
- `Flux2Klein`: 9B parameter model with Qwen3-8b text encoder
- `QwenImage`: 

**Adding New Models:**
```python
# In model_factory.py
elif model_type == 'MyNewModel':
    from diffusers import MyModelTransformer, MyModelPipeline
    
    self.TransformerClass = MyModelTransformer
    self.PipelineClass = MyModelPipeline
    # ... implement load methods
```

### 3. Datasets (`datasets/`)

#### ArXiVParquetDatasetV2

Efficient dataset for pre-computed latents and embeddings:

```python
dataset = dict(
    type='ArXiVParquetDatasetV2',
    base_dir='/path/to/data/',
    parquet_base_path='ArXiV_parquet/Flux2Klein_latents/',
    num_train_examples=100000,
    num_workers=8,
    stat_data=True,
)
```

**Expected Parquet Structure:**
```
ArXiV_parquet/
├── 2015/
│   ├── 2015_rank0_part0.parquet
│   └── 2015/1501.01694/1501.01694_0_flux_latents.npz
├── 2016/
└── ...
```

**Parquet Schema:**
- `paper_id`: Paper identifier
- `image_path`: Original image path
- `caption`: Text caption
- `cache_path`: Path to cached .npz file
- `latent_shape`: Shape of latent
- `prompt_embeds_shape`: Shape of text embeddings
- `bucket_w`, `bucket_h`: Bucket dimensions
- `aspect_ratio`: Aspect ratio category

**NPZ Cache Contents:**
- `latents`: Pre-computed VAE latents (C, H, W)
- `prompt_embeds`: Text embeddings (seq_len, hidden_dim)
- `text_ids`: Position IDs for text (seq_len, 4)

#### ArXiVMixScaleBatchSampler

Custom sampler for mixed-resolution training:

```python
sampler = dict(
    type='ArXiVMixScaleBatchSampler',
    batch_size=2,
    num_replicas=1, ### must be set 1
    rank=0,  # must be set 0
    drop_last=True,
    shuffle=True,
    seed=42,
)
```


### 4. Training Iteration Functions (`train_iteration_funcs/`)

Model-specific training loops implementing the forward pass, loss computation, and backpropagation.

#### Flux2Klein Training Function

Located in `train_iteration_funcs/Flux2Klein_train_iteration_func.py`

**Key Implementation Details:**

1. **Latent Processing**:
   ```python
   # Patchify: (B, 32, H, W) -> (B, 128, H//2, W//2)
   latents = latents.view(B, C, H//2, 2, W//2, 2)
   latents = latents.permute(0, 1, 3, 5, 2, 4)
   latents = latents.reshape(B, C*4, H//2, W//2)
   
   # BatchNorm normalization
   latents = (latents - bn_mean) / bn_std
   ```

2. **Position IDs** (CRITICAL For FLUX.2 klein base 9B):
   ```python
   # Image: [T=0, H_idx, W_idx, L=0]
   img_ids = _prepare_latent_image_ids(
       batch_size=B,
       height=latent_h,
       width=latent_w,
       device=device,
       dtype=dtype,
       already_patchified=True,
   )
   
   # Text: [T=0, H=0, W=0, L=idx]
   txt_ids = batch["text_ids"]  # Pre-computed
   ```

3. **Flow Matching**:
   ```python
   # Add noise
   noisy_input = (1.0 - sigma) * latents + sigma * noise
   
   # Target
   target = noise - latents
   
   # Loss
   loss = MSE(model_pred, target) * weighting
   ```

**Function Signature:**
```python
def Flux2Klein_train_iteration_func_parquet(
    batch,                      # Data batch
    vae,                       # VAE model
    transformer,               # Diffusion transformer
    text_encoding_pipeline,    # Text encoder pipeline
    noise_scheduler_copy,      # Noise scheduler
    accelerator,               # Accelerate instance
    config,                    # Training config
) -> Tuple[torch.Tensor, int]:
    """
    Returns:
        loss: Scalar loss tensor
        batch_size: Batch size for logging
    """
```

**Critical Fixes Applied:**
- ✅ Position ID format corrected to `[T, H, W, L]`
- ✅ Patchify implemented correctly
- ✅ BatchNorm normalization added
- ✅ DDP-wrapped model dtype handling
- ✅ Flow matching target calculation

### 5. Validation Functions (`validation_funcs/`)

#### Generic Validation Function

```python
def generic_validation_func(
    vae,
    transformer,
    text_encoder,
    accelerator,
    scheduler,
    tokenizer,
    args,
):
    """
    Generic validation that works with any model via ModelFactory
    """
```

**Features:**
- Auto-detects model type from config
- Generates validation images
- Logs to WandB
- Saves images to disk

**Configuration:**
```python
validation_prompts = [
    "A diagram showing ...",
    "A flowchart illustrating ...",
]
validation_steps = 500
num_inference_steps = 28
guidance_scale = 3.5
```

### 6. Utilities (`utils/`)

#### Model Utilities (`model_utils.py`)

```python
from OpenSciDraw.utils import (
    unwrap_model,              # Unwrap DDP/compiled models
    get_trainable_params,      # Get trainable parameters
    initialize_models,         # Initialize all models
)
```

#### LoRA Utilities (`lora_utils.py`)

```python
from OpenSciDraw.utils import add_lora_and_load_ckpt_to_models

add_lora_and_load_ckpt_to_models(
    config,
    transformer,
    text_encoder_one,
    text_encoder_two=None,
    pipeline_cls=FluxPipeline,
    filesystem=None,
)
```

#### Training Utilities (`training_utils.py`)

```python
from OpenSciDraw.utils import (
    get_sigmas,                          # Get noise schedule sigmas
    compute_density_for_timestep_sampling,  # Sample timesteps
)
```

## 🆕 Adding New Models

### Step 1: Update Model Factory

Edit `utils/model_factory.py`:

```python
class ModelFactory:
    def __init__(self, config):
        self.config = config
        model_type = getattr(config, 'model_type', 'QwenImage')
        
        # Add your model here
        if model_type == 'MyNewModel':
            from diffusers import MyModelTransformer, MyModelPipeline
            from transformers import MyModelTextEncoder
            
            self.TransformerClass = MyModelTransformer
            self.PipelineClass = MyModelPipeline
            self.TextEncoderClass = MyModelTextEncoder
            
        elif model_type == 'Flux2Klein':
            # ... existing code
    
    def load_transformer(self):
        if self.config.model_type == 'MyNewModel':
            transformer = self.TransformerClass.from_pretrained(
                self.config.pretrained_model_name_or_path,
                subfolder='transformer',
                torch_dtype=self.config.weight_dtype,
            )
            return transformer
        # ... existing code
```

### Step 2: Create Training Function

Create `train_iteration_funcs/MyNewModel_train_iteration_func.py`:

```python
import torch
from OpenSciDraw.registry import TRAIN_ITERATION_FUNCS

@TRAIN_ITERATION_FUNCS.register_module()
def MyNewModel_train_iteration_func(
    batch,
    vae,
    transformer,
    text_encoding_pipeline,
    noise_scheduler_copy,
    accelerator,
    config,
):
    """
    Training iteration for MyNewModel
    """
    # 1. Get text embeddings
    text_embeds = batch["text_embeds"]
    
    # 2. Get latents
    latents = batch["latents"]
    
    # 3. Add noise
    noise = torch.randn_like(latents)
    # ... timestep sampling
    noisy_latents = latents + noise * sigma
    
    # 4. Forward pass
    model_pred = transformer(
        hidden_states=noisy_latents,
        encoder_hidden_states=text_embeds,
        timestep=timesteps,
    )
    
    # 5. Compute loss
    target = noise  # or other targets
    loss = F.mse_loss(model_pred, target)
    
    return loss, batch_size
```

### Step 3: Update Config

```python
# configs/my_new_model_config.py
_base_ = '../base_config.py'

model_type = 'MyNewModel'
pretrained_model_name_or_path = "org/my-new-model"

train_iteration_func_name = 'MyNewModel_train_iteration_func'
```

## 📊 Adding New Datasets

### Step 1: Create Dataset Class

Create `datasets/my_custom_dataset.py`:

```python
import torch
from torch.utils.data import Dataset
from OpenSciDraw.registry import DATASETS

@DATASETS.register_module()
class MyCustomDataset(Dataset):
    def __init__(
        self,
        base_dir,
        num_train_examples=None,
        **kwargs
    ):
        self.base_dir = base_dir
        # Load data index
        self.data = self._load_data()
    
    def _load_data(self):
        # Implement data loading
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return a dictionary with:
        # - 'latents': pre-computed latents or 'pixel_values': raw images
        # - 'text_embeds': pre-computed embeddings or 'caption': raw text
        # - 'bucket_size': (height, width)
        # - 'text_ids': position IDs (if applicable)
        pass
    
    def collate_fn(self, batch):
        # Stack batch items
        return {
            'latents': torch.stack([x['latents'] for x in batch]),
            'text_embeds': torch.stack([x['text_embeds'] for x in batch]),
            'bucket_size': batch[0]['bucket_size'],
        }
```

### Step 2: Register and Use

```python
# In datasets/__init__.py
from .my_custom_dataset import MyCustomDataset

# In config
dataset = dict(
    type='MyCustomDataset',
    base_dir='/path/to/data',
    num_train_examples=10000,
)
```

## 🔬 Custom Training Functions

### Template

```python
from OpenSciDraw.registry import TRAIN_ITERATION_FUNCS
import torch
import torch.nn.functional as F

@TRAIN_ITERATION_FUNCS.register_module()
def custom_train_iteration_func(
    batch,
    vae,
    transformer,
    text_encoding_pipeline,
    noise_scheduler_copy,
    accelerator,
    config,
):
    """
    Custom training iteration
    
    Args:
        batch: Dict with 'latents', 'text_embeds', etc.
        vae: VAE model (may be None if using parquet)
        transformer: Diffusion transformer
        text_encoding_pipeline: Text encoder (may be None)
        noise_scheduler_copy: Noise scheduler
        accelerator: Accelerate instance
        config: Training config
    
    Returns:
        loss: Scalar tensor
        batch_size: Int
    """
    # Get dtype
    dtype = transformer.module.dtype if hasattr(transformer, 'module') else transformer.dtype
    
    # Extract data
    latents = batch["latents"].to(dtype=dtype)
    text_embeds = batch["text_embeds"].to(dtype=dtype)
    bsz = latents.shape[0]
    
    # Generate noise
    noise = torch.randn_like(latents)
    
    # Sample timesteps
    from OpenSciDraw.utils import compute_density_for_timestep_sampling, get_sigmas
    u = compute_density_for_timestep_sampling(
        weighting_scheme=config.weighting_scheme,
        batch_size=bsz,
        logit_mean=config.logit_mean,
        logit_std=config.logit_std,
        mode_scale=config.mode_scale,
    )
    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
    timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)
    
    # Get sigmas
    sigmas = get_sigmas(
        timesteps,
        device=latents.device,
        noise_scheduler_copy=noise_scheduler_copy,
        n_dim=latents.ndim,
        dtype=dtype,
    )
    
    # Add noise
    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
    
    # Forward pass
    model_pred = transformer(
        hidden_states=noisy_latents,
        encoder_hidden_states=text_embeds,
        timestep=timesteps,
    )
    
    # Compute loss
    target = noise - latents  # Flow matching
    loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')
    
    return loss, bsz
```

## 📈 Validation Functions

### Custom Validation

```python
from OpenSciDraw.registry import VALIDATION_FUNCS
import torch

@VALIDATION_FUNCS.register_module()
def custom_validation_func(
    vae,
    transformer,
    text_encoder,
    accelerator,
    scheduler,
    tokenizer,
    args,
):
    """
    Custom validation function
    """
    from OpenSciDraw.utils import ModelFactory
    
    # Get pipeline class
    factory = ModelFactory(args)
    PipelineClass = factory.PipelineClass
    
    # Create pipeline
    pipeline = PipelineClass(
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    pipeline.to(accelerator.device)
    
    # Generate images
    for prompt in args.validation_prompts:
        images = pipeline(
            prompt=prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).images
        
        # Log to WandB
        if args.report_to == 'wandb':
            import wandb
            wandb.log({
                f"validation/{prompt[:20]}": wandb.Image(images[0])
            })
```

## 🐛 Debugging Tips

### 1. Check Data Loading

```python
# In dataset __getitem__
print(f"Loading sample {idx}")
print(f"Latent shape: {latents.shape}")
print(f"Text embeds shape: {text_embeds.shape}")
```

### 2. Verify Model Inputs

```python
# In training function
print(f"Transformer input: {packed_input.shape}")
print(f"Text embeds: {text_embeds.shape}")
print(f"img_ids: {img_ids.shape}")
print(f"txt_ids: {txt_ids.shape}")
```

### 3. Check Position IDs

```python
# Verify format
print(f"img_ids first 3: {img_ids[0, :3]}")
# Should be: [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]]

print(f"txt_ids first 3: {txt_ids[0, :3]}")
# Should be: [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]
```

### 4. Monitor Gradients

```python
# After backward pass
for name, param in transformer.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

## 📚 Best Practices

1. **Always use the registry system** for new components
2. **Test with small data first** before full training
3. **Verify data formats** match model expectations
4. **Use descriptive names** for components
5. **Add docstrings** to all functions
6. **Handle DDP-wrapped models** with `unwrap_model`
7. **Log intermediate values** during debugging
8. **Use type hints** for better code clarity

## 🔗 Related Documentation

- [Main README](../README.md)
- [Configuration Guide](../configs/README.md)
- [Flux2Klein Implementation Details](train_iteration_funcs/Flux2Klein_train_iteration_func.py)

---

**Last Updated:** January 2026
