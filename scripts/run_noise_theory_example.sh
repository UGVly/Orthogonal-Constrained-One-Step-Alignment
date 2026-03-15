#!/usr/bin/env bash
set -euo pipefail

python -m ttt_reward_models.cli_noise_theory \
  --output_dir outputs/orthogonal_gaussian_theory \
  --channels 4 \
  --patch_size 2 \
  --num_samples 65536 \
  --batch_size 4096 \
  --init_scale 0.1
