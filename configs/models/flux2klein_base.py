# Flux2Klein Model Configuration
# Use this as _base_ in your training config to use Flux2Klein model

# ====== Model Type ======
model_type = 'Flux2Klein'

# ====== Model Weights ======
pretrained_model_name_or_path = "black-forest-labs/Flux.2-Klein-9B"  # Example path
# HuggingFace token is now managed centrally
# Priority: Environment variable HF_TOKEN > secrets.py > None
# DO NOT hardcode tokens here - use secrets.py for local dev, or HF_TOKEN env var for AMLT
huggingface_token = None  # Will be resolved from env or secrets.py

# ====== Transformer Config (for registry compatibility) ======
transformer_cfg = dict(
    type='Flux2Transformer2DModel',
)

# ====== VAE Config ======
# Flux2 VAE has different latent structure
vae_scale_factor = 8

# ====== Default LoRA layers for Flux2Klein ======
# These are the typical target modules for Flux2 transformers
lora_layers = "to_k,to_q,to_v,to_out.0"
