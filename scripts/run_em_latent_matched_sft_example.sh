#!/usr/bin/env bash
set -euo pipefail

python -m ttt_reward_models.cli_em_latent_matched_sft \
  --data_root ./data/high_quality_pairs \
  --work_dir ./outputs/em_latent_matched_sft \
  --initial_model_id ./models/sdxl-turbo \
  --num_cycles 2 \
  --assign_steps 40 \
  --assign_lr 5e-4 \
  --train_epochs 1 \
  --train_batch_size 1 \
  --train_lr 1e-5 \
  --latent_loss_weight 1.0 \
  --pixel_l1_weight 0.1 \
  --preserve_latent_weight 0.5 \
  --preserve_pixel_weight 0.05
