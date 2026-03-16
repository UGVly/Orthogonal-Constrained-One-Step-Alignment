#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ttt_reward_models.downloaders import ensure_hpsv2_checkpoint, ensure_imagereward_assets
from ttt_reward_models.paths import get_default_hpsv2_root, get_default_imagereward_root


def main() -> None:
    parser = argparse.ArgumentParser(description='Download reward-model assets into the project-local models directory (legacy third_party_weights paths are still recognized).')
    parser.add_argument('--which', type=str, default='all', choices=['all', 'imagereward', 'hpsv2'])
    parser.add_argument('--hps_version', type=str, default='v2.1', choices=['v2.0', 'v2.1'])
    parser.add_argument('--no_prefer_modelscope', action='store_true')
    args = parser.parse_args()

    if args.which in {'all', 'imagereward'}:
        ckpt, med = ensure_imagereward_assets(
            get_default_imagereward_root(),
            use_modelscope=not args.no_prefer_modelscope,
        )
        print(f'[ImageReward] checkpoint: {ckpt}')
        print(f'[ImageReward] med_config: {med}')

    if args.which in {'all', 'hpsv2'}:
        ckpt = ensure_hpsv2_checkpoint(
            get_default_hpsv2_root(),
            hps_version=args.hps_version,
        )
        print(f'[HPSv2] checkpoint: {ckpt}')


if __name__ == '__main__':
    main()
