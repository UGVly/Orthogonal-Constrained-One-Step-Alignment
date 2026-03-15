import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .downloaders import ensure_hpsv2_checkpoint, stage_hps_root_env
from .paths import get_default_hpsv2_root
from .utils import freeze_module


_GLOBAL_HPS_MODEL: Optional[nn.Module] = None
_GLOBAL_HPS_TOKENIZER = None
_GLOBAL_HPS_KEY: Optional[tuple[str, str]] = None
_GLOBAL_HPS_IMAGE_SIZE: Optional[int] = None


class HPSv2Reward(nn.Module):
    def __init__(
        self,
        prompt: str,
        *,
        hps_version: str = 'v2.1',
        hps_root: Optional[str] = None,
        hps_checkpoint_path: Optional[str] = None,
        auto_download: bool = False,
    ):
        super().__init__()
        self.prompt = prompt
        self.hps_version = hps_version
        self.hps_root = Path(hps_root) if hps_root is not None else get_default_hpsv2_root()

        if auto_download:
            stage_hps_root_env(self.hps_root)

        checkpoint_path = self._resolve_checkpoint(hps_checkpoint_path=hps_checkpoint_path, auto_download=auto_download)

        global _GLOBAL_HPS_MODEL, _GLOBAL_HPS_TOKENIZER, _GLOBAL_HPS_KEY, _GLOBAL_HPS_IMAGE_SIZE
        cache_key = (str(checkpoint_path.resolve()), hps_version)
        if _GLOBAL_HPS_MODEL is None or _GLOBAL_HPS_KEY != cache_key:
            stage_hps_root_env(self.hps_root)
            from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

            model, _, preprocess_val = create_model_and_transforms(
                'ViT-H-14',
                'laion2B-s32B-b79K',
                precision='amp',
                device='cpu',
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False,
            )
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
            state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            _GLOBAL_HPS_MODEL = model
            _GLOBAL_HPS_TOKENIZER = get_tokenizer('ViT-H-14')
            _GLOBAL_HPS_KEY = cache_key
            _GLOBAL_HPS_IMAGE_SIZE = _infer_image_size(preprocess_val)

        self.model = _GLOBAL_HPS_MODEL
        self.tokenizer = _GLOBAL_HPS_TOKENIZER
        self.image_size = _GLOBAL_HPS_IMAGE_SIZE or 224
        freeze_module(self.model)

        text_tokens = self.tokenizer([prompt])
        if isinstance(text_tokens, torch.Tensor):
            self.register_buffer('text_tokens', text_tokens, persistent=False)
        else:
            raise TypeError(f'Unexpected tokenizer output type: {type(text_tokens)!r}')

        self.register_buffer(
            'image_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            'image_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

    def _resolve_checkpoint(self, *, hps_checkpoint_path: Optional[str], auto_download: bool) -> Path:
        if hps_checkpoint_path is not None:
            return Path(hps_checkpoint_path)
        expected = self.hps_root / ('HPS_v2.1_compressed.pt' if self.hps_version == 'v2.1' else 'HPS_v2_compressed.pt')
        if expected.exists():
            return expected
        if auto_download:
            return ensure_hpsv2_checkpoint(self.hps_root, hps_version=self.hps_version)
        raise FileNotFoundError(
            'HPSv2 checkpoint was not found under the project directory. '
            'Run `python scripts/download_reward_assets.py --which hpsv2` first, '
            'or pass --hps_auto_download.'
        )

    def set_device(self, device: str):
        self.to(device)
        self.model.to(device)
        return self

    def preprocess(self, images_01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            images_01,
            size=(self.image_size, self.image_size),
            mode='bicubic',
            align_corners=False,
            antialias=True,
        )
        x = (x - self.image_mean) / self.image_std
        return x

    def forward(self, images_01: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.preprocess(images_01.float())
        outputs = self.model(x, self.text_tokens.to(device=x.device, non_blocking=True))
        image_features = outputs['image_features']
        text_features = outputs['text_features']
        scores = (image_features @ text_features.T).squeeze(-1)
        return {
            'reward': scores,
            'hpsv2': scores,
        }


def _infer_image_size(preprocess_val) -> int:
    transforms = getattr(preprocess_val, 'transforms', [])
    for transform in transforms:
        size = getattr(transform, 'size', None)
        if size is None:
            continue
        if isinstance(size, int):
            return int(size)
        if isinstance(size, (tuple, list)) and len(size) > 0:
            return int(size[0])
    return 224
