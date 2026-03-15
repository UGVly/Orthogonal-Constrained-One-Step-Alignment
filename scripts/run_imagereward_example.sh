#!/usr/bin/env bash
set -euo pipefail

python -m ttt_reward_models.cli_imagereward   --prompt "a cinematic portrait of a girl in soft light, highly detailed"   --imagereward_auto_download   --steps 20   --lr 5e-4   --output_dir outputs/test_time_oft_imagereward_example
