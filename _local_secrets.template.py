# ============================================================
# Secrets Template File
# ============================================================
# Copy this file to 'secrets.py' and fill in your credentials.
# DO NOT commit secrets.py to git!
#
# Usage:
#     cp secrets.template.py secrets.py
#     # Edit secrets.py with your actual credentials
# ============================================================

import os

# ============================================================
# HuggingFace Token
# ============================================================
# For accessing gated models (e.g., Flux.2-Klein-9B)
# Get your token from: https://huggingface.co/settings/tokens
HF_TOKEN = "your_hf_token_here"

# ============================================================
# Weights & Biases (WandB) Configuration
# ============================================================
WANDB_API_KEY = "your_wandb_api_key_here"
WANDB_ENTITY = "your_wandb_entity"
WANDB_PROJECT = "ArXivQwenImage-Run"
WANDB_BASE_URL = "https://api.wandb.ai"  # or your custom wandb server


# ============================================================
# Helper function to get token (with fallback to env vars)
# ============================================================
def get_hf_token():
    """Get HuggingFace token with fallback to environment variable."""
    return os.environ.get("HF_TOKEN", HF_TOKEN)


def get_wandb_api_key():
    """Get WandB API key with fallback to environment variable."""
    return os.environ.get("WANDB_API_KEY", WANDB_API_KEY)


# Export commonly used values
huggingface_token = get_hf_token()
wandb_api_key = get_wandb_api_key()
