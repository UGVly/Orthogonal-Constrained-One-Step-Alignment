#!/usr/bin/env bash
set -euo pipefail

python -m ttt_reward_models.cli_assign_sdxl_sft \
  --prompt "a cinematic portrait of a girl in soft light, highly detailed" \
  --target_image_path ./target.jpg \
  --model_id ./models/sdxl-turbo \
  --output_dir outputs/run_assign_noise_sft \
  --steps 40 \
  --lr 5e-4 \
  --patch_size 2 \
  --latent_loss_weight 1.0 \
  --pixel_l1_weight 0.1
