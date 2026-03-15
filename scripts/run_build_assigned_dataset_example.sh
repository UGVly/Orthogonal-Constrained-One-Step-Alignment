#!/usr/bin/env bash
set -euo pipefail

python -m ttt_reward_models.cli_build_assigned_noise_dataset \
  --data_root ./data/high_quality_pairs \
  --output_root ./outputs/assigned_noise_dataset \
  --model_id ./models/sdxl-turbo \
  --steps 40 \
  --lr 5e-4 \
  --patch_size 2 \
  --latent_loss_weight 1.0 \
  --pixel_l1_weight 0.1
