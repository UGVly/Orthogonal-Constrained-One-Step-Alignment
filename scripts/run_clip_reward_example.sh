#!/usr/bin/env bash
set -e

export PYTHONPATH=./src:${PYTHONPATH}

python -m ttt_reward_models.cli_sdxl_reward \
  --prompt "a cinematic portrait of a girl in soft light, highly detailed" \
  --reward_type hybrid \
  --clip_local_dir /path/to/clip-vit-large-patch14 \
  --aesthetic_ckpt /path/to/aesthetic_head.pth \
  --model_id /path/to/sdxl-turbo \
  --steps 30 \
  --lr 5e-4 \
  --patch_size 2 \
  --output_dir outputs/run_hybrid
