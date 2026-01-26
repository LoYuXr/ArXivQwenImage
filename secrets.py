# ============================================================
# Secrets Configuration File
# ============================================================
# ⚠️  WARNING: This file should NEVER be committed to git!
# ⚠️  Make sure 'secrets.py' is in .gitignore
#
# This file centralizes all sensitive credentials and tokens.
# Other config files should import from here.
#
# Usage in config files:
#     try:
#         from secrets import HF_TOKEN, WANDB_API_KEY
#     except ImportError:
#         import os
#         HF_TOKEN = os.environ.get("HF_TOKEN", None)
#         WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
# ============================================================

import os

# ============================================================
# HuggingFace Token
# ============================================================
# For accessing gated models (e.g., Flux.2-Klein-9B)
# Get your token from: https://huggingface.co/settings/tokens
HF_TOKEN = "***REMOVED***"

# ============================================================
# Weights & Biases (WandB) Configuration
# ============================================================
WANDB_API_KEY = "875d49aadaaa0b7a5d690629d3324b56306709bf"
WANDB_ENTITY = "v-yuxluo"
WANDB_PROJECT = "ArXivQwenImage-Run"
WANDB_BASE_URL = "https://microsoft-research.wandb.io"


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
