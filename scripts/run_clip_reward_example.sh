#!/usr/bin/env bash
set -euo pipefail

python -m ttt_reward_models.cli_sdxl_reward \
  --prompt "a cinematic portrait of a girl in soft light, highly detailed" \
  --reward_type hybrid \
  --clip_local_dir ./models/CLIP-ViT-L-14 \
  --aesthetic_ckpt ./models/Aesthetic/sac+logos+ava1-l14-linearMSE.pth \
  --model_id ./models/sdxl-turbo \
  --steps 30 \
  --lr 5e-4 \
  --patch_size 2 \
  --output_dir outputs/run_hybrid
