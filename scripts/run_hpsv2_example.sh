#!/usr/bin/env bash
set -euo pipefail

python -m ttt_reward_models.cli_hpsv2   --prompt "a cinematic portrait of a girl in soft light, highly detailed"   --hps_auto_download   --hps_version v2.1   --steps 20   --lr 5e-4   --output_dir outputs/test_time_oft_hpsv2_example
