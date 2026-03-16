from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel

from .paths import get_default_aesthetic_ckpt, get_default_clip_root
from .utils import freeze_module


_GLOBAL_CLIP_MODEL: Optional[CLIPModel] = None
_GLOBAL_CLIP_PROCESSOR = None
_GLOBAL_CLIP_SOURCE: Optional[str] = None


class AestheticMLP(nn.Module):
    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CLIPReward(nn.Module):
    def __init__(
        self,
        prompt: str,
        reward_type: str = "clip",
        aesthetic_ckpt: Optional[str] = None,
        clip_local_dir: Optional[str] = None,
    ):
        super().__init__()
        self.reward_type = reward_type

        if clip_local_dir is None:
            default_clip = get_default_clip_root()
            clip_source = str(default_clip) if default_clip.exists() else "openai/clip-vit-large-patch14"
            local_files_only = default_clip.exists()
        else:
            clip_path = Path(clip_local_dir)
            if clip_path.exists():
                clip_source = str(clip_path)
                local_files_only = True
            elif clip_local_dir.startswith((".", "/", "models")):
                raise FileNotFoundError(
                    f"CLIP checkpoint directory was not found: {clip_local_dir}. "
                    "Run `bash scripts/download_models.sh --only CLIP-ViT-L-14` first, or pass a valid Hugging Face model id."
                )
            else:
                clip_source = clip_local_dir
                local_files_only = False

        global _GLOBAL_CLIP_MODEL, _GLOBAL_CLIP_PROCESSOR, _GLOBAL_CLIP_SOURCE
        if _GLOBAL_CLIP_MODEL is None or _GLOBAL_CLIP_SOURCE != clip_source:
            _GLOBAL_CLIP_MODEL = CLIPModel.from_pretrained(
                clip_source,
                local_files_only=local_files_only,
            ).eval()
            _GLOBAL_CLIP_PROCESSOR = AutoProcessor.from_pretrained(
                clip_source,
                local_files_only=local_files_only,
            )
            _GLOBAL_CLIP_SOURCE = clip_source

        self.clip = _GLOBAL_CLIP_MODEL
        self.processor = _GLOBAL_CLIP_PROCESSOR
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.image_processor = getattr(self.processor, "image_processor", None)
        freeze_module(self.clip)

        image_size = None
        if self.image_processor is not None:
            crop_cfg = getattr(self.image_processor, "crop_size", None)
            size_cfg = getattr(self.image_processor, "size", None)
            for cfg in (crop_cfg, size_cfg):
                if isinstance(cfg, dict):
                    if "height" in cfg:
                        image_size = int(cfg["height"])
                        break
                    if "shortest_edge" in cfg:
                        image_size = int(cfg["shortest_edge"])
                        break
                elif isinstance(cfg, int):
                    image_size = int(cfg)
                    break
        if image_size is None:
            image_size = int(self.clip.config.vision_config.image_size)
        self.image_size = image_size
        embed_dim = int(self.clip.config.projection_dim)

        text_inputs = self.tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.register_buffer("text_input_ids", text_inputs["input_ids"], persistent=False)
        self.register_buffer("text_attention_mask", text_inputs["attention_mask"], persistent=False)
        self.register_buffer("text_features", torch.zeros(1, embed_dim), persistent=False)

        image_mean = getattr(self.image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
        image_std = getattr(self.image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std).view(1, 3, 1, 1))

        self.aesthetic_head = None
        if reward_type in {"aesthetic", "hybrid"}:
            if aesthetic_ckpt is None:
                default_aesthetic = get_default_aesthetic_ckpt()
                if default_aesthetic.exists():
                    aesthetic_ckpt = str(default_aesthetic)
                else:
                    raise ValueError(
                        "reward_type='aesthetic' or 'hybrid' requires --aesthetic_ckpt, "
                        f"or a local checkpoint at {default_aesthetic}."
                    )
            elif not Path(aesthetic_ckpt).exists() and aesthetic_ckpt.startswith((".", "/", "models")):
                raise FileNotFoundError(
                    f"Aesthetic checkpoint was not found: {aesthetic_ckpt}. "
                    "Run `bash scripts/download_models.sh --only Aesthetic` first, or pass a valid checkpoint path."
                )

            self.aesthetic_head = AestheticMLP(input_size=embed_dim)
            state = torch.load(aesthetic_ckpt, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            cleaned = {}
            for k, v in state.items():
                nk = k.replace("module.", "")
                cleaned[nk] = v

            self.aesthetic_head.load_state_dict(cleaned, strict=True)
            freeze_module(self.aesthetic_head)

    def set_device(self, device: str):
        self.to(device)
        with torch.no_grad():
            text_features = self.clip.get_text_features(
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
        if not torch.isfinite(images_01).all():
            raise RuntimeError("images_01 already contains NaN/Inf before reward model")

        x = self.preprocess(images_01.float())
        image_features = self.clip.get_image_features(pixel_values=x)
        image_features = F.normalize(image_features.float(), dim=-1)

        if not torch.isfinite(image_features).all():
            raise RuntimeError("image_features contains NaN/Inf")

        clip_cosine = (image_features * self.text_features).sum(dim=-1)
        clip_logit = self.clip.logit_scale.exp().float() * clip_cosine

        if self.reward_type == "clip":
            total = clip_logit
            aesthetic = torch.zeros_like(total)
        elif self.reward_type == "aesthetic":
            aesthetic = self.aesthetic_head(image_features).squeeze(-1)
            total = aesthetic
        else:
            aesthetic = self.aesthetic_head(image_features).squeeze(-1)
            total = aesthetic + 0.25 * clip_cosine

        if not torch.isfinite(total).all():
            raise RuntimeError("reward contains NaN/Inf")

        return {
            "reward": total,
            "aesthetic": aesthetic,
            "clip": clip_logit,
            "clip_cosine": clip_cosine,
        }
