# QwenImage Model Configuration
# Use this as _base_ in your training config to use QwenImage model

# ====== Model Type ======
model_type = 'QwenImage'

# ====== Model Weights ======
pretrained_model_name_or_path = "Qwen/Qwen-Image-2512"
# HuggingFace token is now managed centrally
# Priority: Environment variable HF_TOKEN > secrets.py > None
# DO NOT hardcode tokens here - use secrets.py for local dev, or HF_TOKEN env var for AMLT
huggingface_token = None  # Will be resolved from env or secrets.py

# ====== Transformer Config (for registry compatibility) ======
transformer_cfg = dict(
    type='QwenImageTransformer2DModel',
)

# ====== Default LoRA layers for QwenImage ======
# Full set of target modules for QwenImage transformers
lora_layers = "to_k,to_q,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
