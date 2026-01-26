#!/bin/bash
# ============================================================
# Local Secrets Setup Script
# ============================================================
# This script helps you set up local secrets for development
#
# Usage: ./setup_local_secrets.sh
# ============================================================

echo "============================================================"
echo "Local Secrets Setup"
echo "============================================================"
echo ""
echo "This will create two files:"
echo "  1. secrets.py - Python secrets for config files"
echo "  2. ~/.hf_secrets - Bash secrets for shell scripts"
echo ""

# Check if secrets.py already exists
if [ -f "secrets.py" ]; then
    echo "Warning: secrets.py already exists!"
    read -p "Overwrite? (y/N): " overwrite
    if [ "$overwrite" != "y" ] && [ "$overwrite" != "Y" ]; then
        echo "Keeping existing secrets.py"
        SKIP_PY=true
    fi
fi

echo ""
echo "------------------------------------------------------------"
echo "Enter your credentials (they will NOT be echoed)"
echo "------------------------------------------------------------"
echo ""

# Get HuggingFace token
echo "HuggingFace Token (get from https://huggingface.co/settings/tokens):"
read -s hf_token
echo ""

# Get WandB API key
echo "WandB API Key (get from https://wandb.ai/settings):"
read -s wandb_key
echo ""

# Get WandB entity
read -p "WandB Entity (username/team name): " wandb_entity

# Get WandB base URL
read -p "WandB Base URL [https://api.wandb.ai]: " wandb_url
wandb_url=${wandb_url:-"https://api.wandb.ai"}

echo ""
echo "------------------------------------------------------------"
echo "Creating secrets files..."
echo "------------------------------------------------------------"

# Create secrets.py
if [ "$SKIP_PY" != "true" ]; then
    cat > secrets.py << EOF
# ============================================================
# Secrets Configuration File
# ============================================================
# ⚠️  WARNING: This file should NEVER be committed to git!
# ⚠️  Make sure 'secrets.py' is in .gitignore
# ============================================================

import os

# HuggingFace Token
HF_TOKEN = "${hf_token}"

# Weights & Biases Configuration
WANDB_API_KEY = "${wandb_key}"
WANDB_ENTITY = "${wandb_entity}"
WANDB_PROJECT = "ArXivQwenImage-Run"
WANDB_BASE_URL = "${wandb_url}"


def get_hf_token():
    return os.environ.get("HF_TOKEN", HF_TOKEN)


def get_wandb_api_key():
    return os.environ.get("WANDB_API_KEY", WANDB_API_KEY)


huggingface_token = get_hf_token()
wandb_api_key = get_wandb_api_key()
EOF
    echo "✓ Created secrets.py"
else
    echo "⊘ Skipped secrets.py"
fi

# Create ~/.hf_secrets
cat > ~/.hf_secrets << EOF
# HuggingFace and WandB secrets for shell scripts
# Source this file in your scripts: source ~/.hf_secrets
export HF_TOKEN="${hf_token}"
export HUGGING_FACE_HUB_TOKEN="${hf_token}"
export WANDB_API_KEY="${wandb_key}"
export WANDB_ENTITY="${wandb_entity}"
export WANDB_BASE_URL="${wandb_url}"
EOF
chmod 600 ~/.hf_secrets
echo "✓ Created ~/.hf_secrets (permissions: 600)"

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "The following files were created:"
echo "  - secrets.py (for Python configs)"
echo "  - ~/.hf_secrets (for shell scripts)"
echo ""
echo "These files are already in .gitignore and will NOT be committed."
echo ""
echo "To use in shell scripts, add: source ~/.hf_secrets"
echo "To use in Python, import from secrets.py or use token_helper.py"
echo "============================================================"
