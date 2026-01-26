#!/bin/bash
# ============================================================
# AMLT Secrets Setup Script
# ============================================================
# This script helps you set up AMLT secrets for HF_TOKEN and WANDB_API_KEY
# Run this once before submitting AMLT jobs
#
# Usage: ./setup_amlt_secrets.sh
# ============================================================

echo "============================================================"
echo "AMLT Secrets Setup"
echo "============================================================"
echo ""
echo "This script will help you set up AMLT secrets for:"
echo "  - HF_TOKEN (HuggingFace token)"
echo "  - WANDB_API_KEY (Weights & Biases API key)"
echo ""
echo "These secrets will be stored securely in Azure and referenced"
echo "in your AMLT yaml files with \$\${{secrets.SECRET_NAME}}"
echo ""

# Check if amlt is installed
if ! command -v amlt &> /dev/null; then
    echo "Error: amlt command not found. Please install Azure ML Tools first."
    echo "  pip install azure-ml-tools"
    exit 1
fi

echo "------------------------------------------------------------"
echo "Step 1: Set HuggingFace Token"
echo "------------------------------------------------------------"
echo "Get your token from: https://huggingface.co/settings/tokens"
echo ""
read -p "Enter your HuggingFace token (or press Enter to skip): " hf_token

if [ -n "$hf_token" ]; then
    echo "Setting HF_TOKEN secret..."
    amlt secret set HF_TOKEN "$hf_token"
    if [ $? -eq 0 ]; then
        echo "✓ HF_TOKEN secret set successfully"
    else
        echo "✗ Failed to set HF_TOKEN secret"
    fi
else
    echo "Skipping HF_TOKEN..."
fi

echo ""
echo "------------------------------------------------------------"
echo "Step 2: Set WandB API Key"
echo "------------------------------------------------------------"
echo "Get your key from: https://wandb.ai/settings"
echo ""
read -p "Enter your WandB API key (or press Enter to skip): " wandb_key

if [ -n "$wandb_key" ]; then
    echo "Setting WANDB_API_KEY secret..."
    amlt secret set WANDB_API_KEY "$wandb_key"
    if [ $? -eq 0 ]; then
        echo "✓ WANDB_API_KEY secret set successfully"
    else
        echo "✗ Failed to set WANDB_API_KEY secret"
    fi
else
    echo "Skipping WANDB_API_KEY..."
fi

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Now update your AMLT yaml files to use secrets:"
echo ""
echo "  BEFORE (insecure, don't commit to git):"
echo "    - export HF_TOKEN=hf_xxxxxxxxxx"
echo "    - export WANDB_API_KEY=xxxxxxxxxx"
echo ""
echo "  AFTER (secure, safe to commit):"
echo "    - export HF_TOKEN=\$\${{secrets.HF_TOKEN}}"
echo "    - export WANDB_API_KEY=\$\${{secrets.WANDB_API_KEY}}"
echo ""
echo "See amlt/template_secure_tokens.yaml for a full example."
echo "============================================================"
