from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .downloaders import ensure_imagereward_assets, find_imagereward_assets
from .paths import get_default_imagereward_root
from .utils import freeze_module


_GLOBAL_IMAGEREWARD_MODEL: Optional[nn.Module] = None
_GLOBAL_IMAGEREWARD_KEY: Optional[tuple[str, str]] = None


class ImageRewardReward(nn.Module):
    def __init__(
        self,
        prompt: str,
        *,
        imagereward_root: Optional[str] = None,
        imagereward_model_path: Optional[str] = None,
        imagereward_med_config_path: Optional[str] = None,
        auto_download: bool = False,
        prefer_modelscope: bool = True,
    ):
        super().__init__()
        self.prompt = prompt
        self.imagereward_root = Path(imagereward_root) if imagereward_root is not None else get_default_imagereward_root()

        model_path, med_path = self._resolve_paths(
            imagereward_model_path=imagereward_model_path,
            imagereward_med_config_path=imagereward_med_config_path,
            auto_download=auto_download,
            prefer_modelscope=prefer_modelscope,
        )

        global _GLOBAL_IMAGEREWARD_MODEL, _GLOBAL_IMAGEREWARD_KEY
        cache_key = (str(model_path.resolve()), str(med_path.resolve()))
        if _GLOBAL_IMAGEREWARD_MODEL is None or _GLOBAL_IMAGEREWARD_KEY != cache_key:
            import ImageReward as RM

            _GLOBAL_IMAGEREWARD_MODEL = RM.load(
                name=str(model_path),
                device='cpu',
                download_root=str(self.imagereward_root),
                med_config=str(med_path),
            ).eval()
            _GLOBAL_IMAGEREWARD_KEY = cache_key

        self.model = _GLOBAL_IMAGEREWARD_MODEL
        freeze_module(self.model)

        text_inputs = self.model.blip.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=35,
            return_tensors='pt',
        )
        self.register_buffer('text_input_ids', text_inputs['input_ids'], persistent=False)
        self.register_buffer('text_attention_mask', text_inputs['attention_mask'], persistent=False)
        self.register_buffer(
            'image_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            'image_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )
        self.image_size = 224

    def _resolve_paths(
        self,
        *,
        imagereward_model_path: Optional[str],
        imagereward_med_config_path: Optional[str],
        auto_download: bool,
        prefer_modelscope: bool,
    ) -> tuple[Path, Path]:
        if imagereward_model_path is not None and imagereward_med_config_path is not None:
            return Path(imagereward_model_path), Path(imagereward_med_config_path)

        found_model, found_med = find_imagereward_assets(self.imagereward_root)
        if found_model is not None and found_med is not None:
            return found_model, found_med

        if auto_download:
            return ensure_imagereward_assets(self.imagereward_root, use_modelscope=prefer_modelscope)

        raise FileNotFoundError(
            'ImageReward weights were not found under the project directory. '
            'Run `python scripts/download_reward_assets.py --which imagereward` first, '
            'or pass --imagereward_auto_download.'
        )

    def set_device(self, device: str):
        self.to(device)
        self.model.to(device)
        self.model.device = device
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
        rewards = self.model.score_gard(
            self.text_input_ids.to(device=x.device),
            self.text_attention_mask.to(device=x.device),
            x,
        ).squeeze(-1)
        return {
            'reward': rewards,
            'imagereward': rewards,
        }
