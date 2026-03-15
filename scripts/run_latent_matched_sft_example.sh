#!/usr/bin/env bash
set -euo pipefail

python -m ttt_reward_models.cli_train_latent_matched_sft \
  --manifest_path ./outputs/assigned_noise_dataset/manifest.jsonl \
  --model_id ./models/sdxl-turbo \
  --output_dir ./outputs/latent_matched_sft \
  --epochs 2 \
  --batch_size 1 \
  --lr 1e-5 \
  --latent_loss_weight 1.0 \
  --pixel_l1_weight 0.1 \
  --preserve_latent_weight 0.5 \
  --preserve_pixel_weight 0.05
