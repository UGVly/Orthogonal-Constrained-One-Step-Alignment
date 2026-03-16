from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from .paths import get_default_pickscore_root
from .utils import freeze_module


_GLOBAL_PICKSCORE_MODEL: Optional[nn.Module] = None
_GLOBAL_PICKSCORE_PROCESSOR: Optional[AutoProcessor] = None
_GLOBAL_PICKSCORE_ID: Optional[str] = None


class PickScoreReward(nn.Module):
    def __init__(
        self,
        prompt: str,
        pickscore_model_id: str = "yuvalkirstain/PickScore_v1",
    ):
        super().__init__()
        self.prompt = prompt

        if pickscore_model_id == "yuvalkirstain/PickScore_v1":
            default_local = get_default_pickscore_root()
            if default_local.exists():
                pickscore_model_id = str(default_local)
        elif not Path(pickscore_model_id).exists() and pickscore_model_id.startswith((".", "/", "models")):
            raise FileNotFoundError(
                f"PickScore checkpoint directory was not found: {pickscore_model_id}. "
                "Run `bash scripts/download_models.sh --only PickScore_v1` first, or pass a valid Hugging Face repo id."
            )
        self.pickscore_model_id = pickscore_model_id
        local_files_only = Path(self.pickscore_model_id).exists()

        global _GLOBAL_PICKSCORE_MODEL, _GLOBAL_PICKSCORE_PROCESSOR, _GLOBAL_PICKSCORE_ID
        if _GLOBAL_PICKSCORE_MODEL is None or _GLOBAL_PICKSCORE_ID != self.pickscore_model_id:
            _GLOBAL_PICKSCORE_MODEL = AutoModel.from_pretrained(
                self.pickscore_model_id,
                local_files_only=local_files_only,
            ).eval()
            _GLOBAL_PICKSCORE_PROCESSOR = AutoProcessor.from_pretrained(
                self.pickscore_model_id,
                local_files_only=local_files_only,
            )
            _GLOBAL_PICKSCORE_ID = self.pickscore_model_id

        self.model = _GLOBAL_PICKSCORE_MODEL
        self.processor = _GLOBAL_PICKSCORE_PROCESSOR
        freeze_module(self.model)

        image_processor = self.processor.image_processor
        size_cfg = getattr(image_processor, "crop_size", None) or image_processor.size
        if isinstance(size_cfg, dict):
            if "shortest_edge" in size_cfg:
                self.image_size = int(size_cfg["shortest_edge"])
            elif "height" in size_cfg:
                self.image_size = int(size_cfg["height"])
            else:
                self.image_size = 224
        else:
            self.image_size = int(size_cfg)

        image_mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
        image_std = getattr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std).view(1, 3, 1, 1))

        text_inputs = self.processor(
            text=[prompt],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.register_buffer("text_input_ids", text_inputs["input_ids"], persistent=False)
        self.register_buffer("text_attention_mask", text_inputs["attention_mask"], persistent=False)
        embed_dim = int(getattr(self.model.config, "projection_dim", 768))
        self.register_buffer("text_features", torch.zeros(1, embed_dim), persistent=False)

    def set_device(self, device: str):
        self.to(device)
        with torch.no_grad():
            text_features = self.model.get_text_features(
                input_ids=self.text_input_ids.to(device),
                attention_mask=self.text_attention_mask.to(device),
            )
            text_features = F.normalize(text_features.float(), dim=-1)
            self.text_features.copy_(text_features)
        return self

    def preprocess(self, images_01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            images_01,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        x = (x - self.image_mean) / self.image_std
        return x

    def forward(self, images_01: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.preprocess(images_01.float())
        image_features = self.model.get_image_features(pixel_values=x)
        image_features = F.normalize(image_features.float(), dim=-1)
        cosine = (image_features * self.text_features).sum(dim=-1)
        logit_scale = getattr(self.model, "logit_scale", None)
        if logit_scale is not None:
            pickscore = logit_scale.exp().float() * cosine
        else:
            pickscore = cosine
        return {
            "reward": pickscore,
            "pickscore": pickscore,
            "pickscore_cosine": cosine,
        }
