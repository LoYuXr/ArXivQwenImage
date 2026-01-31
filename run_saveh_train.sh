#!/bin/bash
# Flux2Klein 9B Full Fine-tuning with save_h format
# Dataset: 2015 + 2022 + 2023 (~15k+ samples)
# Training: 10000 steps, GA=2, validation every 250 steps, checkpoint every 1000 steps

cd /home/v-yuxluo/WORK_local/ArXivQwenImage
source /home/v-yuxluo/miniconda3/etc/profile.d/conda.sh
conda activate flux2

# Use GA=2 config for better throughput
accelerate launch --config_file accelerate_cfg/deepspeed_zero2_bf16_ga2.yaml \
    train_OpenSciDraw_fulltune.py configs/260130/flux2klein_saveh_local_10k.py \
    2>&1 | tee logs/flux2klein_saveh_10k.log
