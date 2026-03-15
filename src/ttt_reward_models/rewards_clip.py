from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from .utils import freeze_module


_GLOBAL_CLIP_MODEL: Optional[CLIPModel] = None
_GLOBAL_CLIP_TOKENIZER: Optional[CLIPTokenizer] = None
_GLOBAL_CLIP_LOCAL_DIR: Optional[str] = None


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
            raise ValueError("Please pass --clip_local_dir for local CLIP loading.")

        global _GLOBAL_CLIP_MODEL, _GLOBAL_CLIP_TOKENIZER, _GLOBAL_CLIP_LOCAL_DIR
        if _GLOBAL_CLIP_MODEL is None or _GLOBAL_CLIP_LOCAL_DIR != clip_local_dir:
            _GLOBAL_CLIP_MODEL = CLIPModel.from_pretrained(
                clip_local_dir,
                local_files_only=True,
            ).eval()
            _GLOBAL_CLIP_TOKENIZER = CLIPTokenizer.from_pretrained(
                clip_local_dir,
                local_files_only=True,
            )
            _GLOBAL_CLIP_LOCAL_DIR = clip_local_dir

        self.clip = _GLOBAL_CLIP_MODEL
        self.tokenizer = _GLOBAL_CLIP_TOKENIZER
        freeze_module(self.clip)

        self.image_size = self.clip.config.vision_config.image_size
        embed_dim = self.clip.config.projection_dim

        text_inputs = self.tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.register_buffer("text_input_ids", text_inputs["input_ids"], persistent=False)
        self.register_buffer("text_attention_mask", text_inputs["attention_mask"], persistent=False)
        self.register_buffer("text_features", torch.zeros(1, embed_dim), persistent=False)

        self.register_buffer(
            "image_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

        self.aesthetic_head = None
        if reward_type in {"aesthetic", "hybrid"}:
            if aesthetic_ckpt is None:
                raise ValueError("reward_type='aesthetic' or 'hybrid' requires --aesthetic_ckpt.")

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
            text_outputs = self.clip.text_model(
                input_ids=self.text_input_ids.to(device),
                attention_mask=self.text_attention_mask.to(device),
                return_dict=True,
            )
            text_features = self.clip.text_projection(text_outputs.pooler_output)
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
        vision_outputs = self.clip.vision_model(
            pixel_values=x,
            return_dict=True,
        )
        image_features = self.clip.visual_projection(vision_outputs.pooler_output)
        image_features = F.normalize(image_features.float(), dim=-1)

        if not torch.isfinite(image_features).all():
            raise RuntimeError("image_features contains NaN/Inf")

        clip_score = (image_features * self.text_features).sum(dim=-1)

        if self.reward_type == "clip":
            total = clip_score
            aesthetic = torch.zeros_like(total)
        elif self.reward_type == "aesthetic":
            aesthetic = self.aesthetic_head(image_features).squeeze(-1)
            total = aesthetic
        else:
            aesthetic = self.aesthetic_head(image_features).squeeze(-1)
            total = aesthetic + 0.25 * clip_score

        if not torch.isfinite(total).all():
            raise RuntimeError("reward contains NaN/Inf")

        return {
            "reward": total,
            "aesthetic": aesthetic,
            "clip": clip_score,
        }
