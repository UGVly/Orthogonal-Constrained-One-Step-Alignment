#!/usr/bin/env bash
set -euo pipefail

python -m ttt_reward_models.cli_hpsv2 \
  --prompt "a cinematic portrait of a girl in soft light, highly detailed" \
  --model_id ./models/sdxl-turbo \
  --hps_checkpoint_path ./models/HPSv2/HPS_v2.1_compressed.pt \
  --hps_version v2.1 \
  --steps 20 \
  --lr 5e-4 \
  --output_dir outputs/test_time_oft_hpsv2_example
