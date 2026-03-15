#!/usr/bin/env bash
set -e

export PYTHONPATH=./src:${PYTHONPATH}

python -m ttt_reward_models.cli_batch_pickscore \
  --prompt_file examples/prompts.txt \
  --output_dir outputs/prompt_batch \
  --model_id /path/to/sdxl-turbo \
  --pickscore_model_id yuvalkirstain/PickScore_v1 \
  --steps 30 \
  --lr 5e-4 \
  --patch_size 2
