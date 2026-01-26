# Secrets Management Guide

This document explains how to manage sensitive credentials (HuggingFace tokens, WandB API keys, etc.) securely in this project.

## Overview

We use a **centralized secrets management** approach with the following priority hierarchy:

1. **Environment Variables** (highest priority) - Best for AMLT/cloud deployments
2. **_local_secrets.py** - For local development
3. **None/Default** - Falls back gracefully

## Quick Start

### For Local Development

Run the setup script:

```bash
./setup_local_secrets.sh
```

This creates:
- `_local_secrets.py` - Python secrets for config files
- `~/.hf_secrets` - Bash secrets for shell scripts

### For AMLT/Cloud Deployment

1. Set up AMLT secrets:
```bash
./setup_amlt_secrets.sh
# Or manually:
amlt secret set HF_TOKEN "your_hf_token"
amlt secret set WANDB_API_KEY "your_wandb_key"
```

2. Use secrets in your AMLT yaml:
```yaml
command:
  - export HF_TOKEN=$${{secrets.HF_TOKEN}}
  - export WANDB_API_KEY=$${{secrets.WANDB_API_KEY}}
```

See `amlt/template_secure_tokens.yaml` for a full example.

## File Structure

```
├── _local_secrets.py              # Your local secrets (git-ignored)
├── secrets.template.py     # Template file (safe to commit)
├── setup_local_secrets.sh  # Script to create _local_secrets.py
├── setup_amlt_secrets.sh   # Script to set AMLT secrets
├── ~/.hf_secrets          # Bash secrets for shell scripts
└── OpenSciDraw/utils/
    └── token_helper.py    # Centralized token management
```

## How It Works

### In Python Config Files

The `base_config.py` automatically resolves tokens:

```python
# Priority: HF_TOKEN env var > _local_secrets.py > None
try:
    from secrets import HF_TOKEN as _HF_TOKEN
    huggingface_token = os.environ.get("HF_TOKEN", _HF_TOKEN)
except ImportError:
    huggingface_token = os.environ.get("HF_TOKEN", None)
```

### In Python Code

Use the `token_helper` module:

```python
from OpenSciDraw.utils.token_helper import get_hf_token, get_wandb_config

# Get HF token with fallback hierarchy
token = get_hf_token()

# Get all WandB config
wandb_cfg = get_wandb_config()
```

### In Shell Scripts

Source the secrets file:

```bash
if [ -f ~/.hf_secrets ]; then
    source ~/.hf_secrets
fi

# Now $HF_TOKEN and $WANDB_API_KEY are available
```

## Security Best Practices

1. **Never commit secrets** - All secrets files are in `.gitignore`
2. **Use environment variables in CI/CD** - Set `HF_TOKEN` in your pipeline
3. **Use AMLT secrets for cloud jobs** - More secure than hardcoding
4. **Rotate tokens regularly** - If a token is exposed, regenerate it

## Migration from Hardcoded Tokens

If you had hardcoded tokens in your config files:

1. Run `./setup_local_secrets.sh` to create proper secrets files
2. The old hardcoded tokens have been replaced with `huggingface_token = None`
3. The code will automatically use `_local_secrets.py` or environment variables

## Troubleshooting

### "Token not found" errors

1. Check if `_local_secrets.py` exists:
   ```bash
   ls -la _local_secrets.py
   ```

2. Check environment variable:
   ```bash
   echo $HF_TOKEN
   ```

3. Run the debug helper:
   ```python
   from OpenSciDraw.utils.token_helper import print_token_status
   print_token_status()
   ```

### AMLT job fails with auth error

1. Verify secrets are set:
   ```bash
   amlt secret list
   ```

2. Make sure your yaml uses correct syntax:
   ```yaml
   - export HF_TOKEN=$${{secrets.HF_TOKEN}}
   ```
   Note: Use `$$` (double dollar sign) for AMLT secret references.

## Files That Are Git-Ignored

The following patterns are in `.gitignore`:

```
_local_secrets.py
secrets_*.py
!secrets.template.py
*.secret
.hf_secrets
*_secrets.sh
**/hf_token*
**/wandb_key*
.env
.env.*
```
