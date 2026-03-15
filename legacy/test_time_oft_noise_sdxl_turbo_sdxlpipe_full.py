import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from diffusers import StableDiffusionXLPipeline
from transformers import CLIPModel, CLIPTokenizer
import time
from tqdm import tqdm
# -------------------------------------------------
# Example
'''
python test_time_oft_noise_sdxl_turbo_sdxlpipe_full.py \
  --prompt "a cinematic portrait of a girl in soft light, highly detailed" \
  --reward_type hybrid \
  --clip_local_dir /home/jiangzhou/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41 \
  --aesthetic_ckpt /home/jiangzhou/CODE/Text2ImageProject/OneStepAlign/TestTimeTrain/sac+logos+ava1-l14-linearMSE.pth \
  --steps 30 \
  --lr 5e-4 \
  --patch_size 2 \
  --output_dir outputs/run_debug
'''
# -------------------------------------------------


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def get_device(device_arg: Optional[str]) -> str:
    if device_arg is not None:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Global caches for heavy models
# -----------------------------

_GLOBAL_PIPE: Optional[StableDiffusionXLPipeline] = None
_GLOBAL_PIPE_KEY: Optional[Tuple[str, str, str]] = None

_GLOBAL_CLIP_MODEL: Optional[CLIPModel] = None
_GLOBAL_CLIP_TOKENIZER: Optional[CLIPTokenizer] = None
_GLOBAL_CLIP_LOCAL_DIR: Optional[str] = None


# -----------------------------
# Patch-wise orthogonal epsilon transform
# -----------------------------

class PatchOrthogonalNoise(nn.Module):
    r"""
    Shared orthogonal transform applied on flattened latent patches.

    For epsilon in R^{B x C x H x W}, split into non-overlapping patches of size p x p,
    flatten each patch to dimension D = C * p * p, then multiply by a learnable
    orthogonal matrix Q in R^{D x D}.

    Q = exp(S), where S is skew-symmetric, so Q is exactly orthogonal.
    This preserves the l2 energy of each patch.
    """

    def __init__(self, channels: int = 4, patch_size: int = 2, init_scale: float = 1e-4):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.block_dim = channels * patch_size * patch_size

        raw = torch.zeros(self.block_dim, self.block_dim)
        raw += init_scale * torch.randn_like(raw)
        self.raw = nn.Parameter(raw)

    def orthogonal_matrix(self) -> torch.Tensor:
        skew = self.raw - self.raw.transpose(0, 1)
        q = torch.matrix_exp(skew)
        return q

    def forward(self, eps: torch.Tensor) -> torch.Tensor:
        b, c, h, w = eps.shape
        p = self.patch_size
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {c}.")
        if h % p != 0 or w % p != 0:
            raise ValueError(f"Latent size {(h, w)} must be divisible by patch_size={p}.")

        q = self.orthogonal_matrix().to(dtype=eps.dtype, device=eps.device)

        x = eps.view(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()       # [B, H/p, W/p, C, p, p]
        x = x.view(b, -1, self.block_dim)                  # [B, Npatch, D]
        x = x @ q.T                                        # shared orthogonal map
        x = x.view(b, h // p, w // p, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, h, w)
        return x


# -----------------------------
# Noise spectrum diagnostics
# -----------------------------

def noise_to_patch_matrix(noise: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert BCHW noise into [N_patch_total, D] patch matrix,
    where D = C * patch_size * patch_size.
    """
    b, c, h, w = noise.shape
    p = patch_size
    if h % p != 0 or w % p != 0:
        raise ValueError(f"Noise shape {(h, w)} must be divisible by patch_size={p}.")

    x = noise.view(b, c, h // p, p, w // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.view(-1, c * p * p)
    return x


def covariance_eigenvalues_from_noise(
    noise: torch.Tensor,
    patch_size: int = 2,
    center: bool = True,
) -> torch.Tensor:
    """
    We define the 'eigenvalues of noise' as the eigenvalues of the patch-feature covariance.
    For a BCHW noise tensor, first flatten non-overlapping patches into vectors,
    then compute covariance over patch vectors, and finally eigvalsh(cov).
    """
    x = noise_to_patch_matrix(noise.detach(), patch_size=patch_size).to(dtype=torch.float64, device="cpu")
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    n = x.shape[0]
    cov = (x.T @ x) / max(n - 1, 1)
    cov = 0.5 * (cov + cov.T)
    eigvals = torch.linalg.eigvalsh(cov)
    return eigvals


def summarize_noise(noise: torch.Tensor, name: str) -> None:
    x = noise.detach().float()
    # print(
    #     f"{name}: mean={x.mean().item():+.6f}, std={x.std().item():.6f}, "
    #     f"min={x.min().item():+.6f}, max={x.max().item():+.6f}, l2={x.norm().item():.6f}"
    # )


def print_eig_report(step: int, base_eigs: torch.Tensor, cur_eigs: torch.Tensor) -> None:
    diff = (cur_eigs - base_eigs).abs()
    # print(f"[step {step:03d}] eig max|Δ|={diff.max().item():.8e}, mean|Δ|={diff.mean().item():.8e}")
    # print(f"[step {step:03d}] base eigs = {[round(v, 8) for v in base_eigs.tolist()]}")
    # print(f"[step {step:03d}] curr eigs = {[round(v, 8) for v in cur_eigs.tolist()]}")
    # print(f"[step {step:03d}] |Δeig|   = {[round(v, 8) for v in diff.tolist()]}")


def print_q_spectrum(step: int, q: torch.Tensor) -> None:
    svals = torch.linalg.svdvals(q.detach().float().cpu())
    # print(
    #     f"[step {step:03d}] Q singular values min/max = "
    #     f"{svals.min().item():.8f} / {svals.max().item():.8f}"
    # )


# -----------------------------
# Reward model
# -----------------------------

class AestheticMLP(nn.Module):
    """
    Compatible with sac+logos+ava1-l14-linearMSE.pth from improved aesthetic predictor.
    """

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

        # Standard CLIP normalization.
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
            # print("[aesthetic] loaded ok")
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
        else:  # hybrid
            aesthetic = self.aesthetic_head(image_features).squeeze(-1)
            total = aesthetic + 0.25 * clip_score

        if not torch.isfinite(total).all():
            raise RuntimeError("reward contains NaN/Inf")

        return {
            "reward": total,
            "aesthetic": aesthetic,
            "clip": clip_score,
        }


# -----------------------------
# SDXL Turbo differentiable one-step forward
# -----------------------------

@dataclass
class PromptCond:
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    add_time_ids: torch.Tensor


class OneStepSDXLTurbo(nn.Module):
    def __init__(self, pipe, height: int = 512, width: int = 512):
        super().__init__()
        self.pipe = pipe
        self.height = height
        self.width = width
        self.vae_scale_factor = pipe.vae_scale_factor
        self.latent_h = height // self.vae_scale_factor
        self.latent_w = width // self.vae_scale_factor

    @property
    def device(self):
        return self.pipe._execution_device

    @property
    def model_dtype(self):
        return self.pipe.unet.dtype

    def build_prompt_cond(self, prompt: str, device: str, batch_size: int = 1) -> PromptCond:
        pe, _, pooled, _ = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=batch_size,
            do_classifier_free_guidance=False,
        )

        text_encoder_projection_dim = None
        if getattr(self.pipe, "text_encoder_2", None) is not None:
            text_encoder_projection_dim = getattr(self.pipe.text_encoder_2.config, "projection_dim", None)

        add_time_ids = self.pipe._get_add_time_ids(
            (self.height, self.width),
            (0, 0),
            (self.height, self.width),
            dtype=pe.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.to(device)
        add_time_ids = add_time_ids.repeat(batch_size, 1)

        return PromptCond(
            prompt_embeds=pe,
            pooled_prompt_embeds=pooled,
            add_time_ids=add_time_ids,
        )

    def sample_base_noise(self, batch_size: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        shape = (batch_size, self.pipe.unet.config.in_channels, self.latent_h, self.latent_w)
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)
        latents = latents * float(self.pipe.scheduler.init_noise_sigma)
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae = self.pipe.vae
        latents = latents.to(device=vae.device, dtype=torch.float32)
        latents = latents / vae.config.scaling_factor

        if not torch.isfinite(latents).all():
            raise RuntimeError("NaN/Inf appears before VAE decode")

        if latents.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                image = vae.decode(latents, return_dict=False)[0]
        else:
            image = vae.decode(latents, return_dict=False)[0]

        if not torch.isfinite(image).all():
            raise RuntimeError("NaN/Inf returned directly by vae.decode")

        image = (image.float() / 2 + 0.5).clamp(0, 1)
        return image

    def forward(self, latents: torch.Tensor, cond: PromptCond) -> Tuple[torch.Tensor, torch.Tensor]:
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(1, device=latents.device)
        t = scheduler.timesteps[0]

        latent_model_input = scheduler.scale_model_input(latents.to(self.model_dtype), t)

        noise_pred = self.pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=cond.prompt_embeds,
            added_cond_kwargs={
                "text_embeds": cond.pooled_prompt_embeds,
                "time_ids": cond.add_time_ids,
            },
            return_dict=False,
        )[0]

        denoised = scheduler.step(
            noise_pred.float(),
            t,
            latents.float(),
            return_dict=False,
        )[0]

        if not torch.isfinite(denoised).all():
            raise RuntimeError("NaN/Inf appears right after scheduler.step")

        images = self.decode_latents(denoised.float())

        if not torch.isfinite(images).all():
            raise RuntimeError("NaN/Inf appears after VAE decode")

        return images, denoised


# -----------------------------
# Main optimization loop
# -----------------------------

def save_grid(images: torch.Tensor, path: str) -> None:
    nrow = max(1, int(math.sqrt(images.shape[0])))
    save_image(images.detach().cpu(), path, nrow=nrow)


def run_test_time_oft(
    prompt: str,
    *,
    model_id: str = "/home/jiangzhou/.cache/modelscope/hub/models/AI-ModelScope/sdxl-turbo/",
    output_dir: Optional[str] = "outputs/test_time_oft_noise",
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
    steps: int = 30,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    noise_l2_weight: float = 1e-3,
    patch_size: int = 2,
    reward_type: str = "clip",
    aesthetic_ckpt: Optional[str] = None,
    clip_local_dir: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 42,
    variant: str = "fp16",
    save_every: int = 1,
    print_eigs_every: int = 1,
    save_outputs: bool = True,
) -> Tuple[List[torch.Tensor], Dict[str, object]]:
    """
    对给定 prompt 做 test-time 正交噪声优化，返回最佳图像列表和对应的 reward 字典。

    Args:
        prompt: 文本提示
        其他参数与命令行参数一致；save_outputs 为 False 时不写盘（仅返回结果）。

    Returns:
        image_list: 最佳步对应的图像列表，每项为 [C, H, W] 的 tensor，范围 [0, 1]
        reward_dict: 同时包含 best / last / each-step 历史分数。
    """
    if output_dir and save_outputs:
        ensure_dir(output_dir)
    set_seed(seed)
    dev = get_device(device)
    torch.set_grad_enabled(True)

    # Use cached SDXL pipeline to avoid re-loading weights every call.
    global _GLOBAL_PIPE, _GLOBAL_PIPE_KEY
    pipe_key = (model_id, dev, variant if dev.startswith("cuda") else "none")
    if _GLOBAL_PIPE is None or _GLOBAL_PIPE_KEY != pipe_key:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if dev.startswith("cuda") else torch.float32,
            variant=variant if dev.startswith("cuda") else None,
        )
        pipe = pipe.to(dev)
        pipe.scheduler = pipe.scheduler.__class__.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
        )

        # Freeze backbone, train only epsilon transform.
        freeze_module(pipe.unet)
        freeze_module(pipe.vae)
        if getattr(pipe, "text_encoder", None) is not None:
            freeze_module(pipe.text_encoder)
        if getattr(pipe, "text_encoder_2", None) is not None:
            freeze_module(pipe.text_encoder_2)
        # Keep VAE in fp32 for SDXL stability.
        pipe.vae.to(dtype=torch.float32)

        _GLOBAL_PIPE = pipe
        _GLOBAL_PIPE_KEY = pipe_key
    pipe = _GLOBAL_PIPE

    model = OneStepSDXLTurbo(pipe, height=height, width=width)
    cond = model.build_prompt_cond(prompt, dev, batch_size=batch_size)

    reward_model = CLIPReward(
        prompt=prompt,
        reward_type=reward_type,
        aesthetic_ckpt=aesthetic_ckpt,
        clip_local_dir=clip_local_dir,
    ).set_device(dev)

    adapter = PatchOrthogonalNoise(
        channels=pipe.unet.config.in_channels,
        patch_size=patch_size,
    ).to(dev)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)

    gen_device = dev if dev.startswith("cuda") else "cpu"
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)
    base_noise = model.sample_base_noise(batch_size, generator=generator)

    base_eigs = covariance_eigenvalues_from_noise(base_noise, patch_size=patch_size)
    summarize_noise(base_noise, "base_noise")
    # print(f"[base] eigs = {[round(v, 8) for v in base_eigs.tolist()]}")

    with torch.no_grad():
        init_images, _ = model(base_noise.to(dev), cond)
        if output_dir and save_outputs:
            save_grid(init_images, os.path.join(output_dir, "init.png"))

    best_reward = -1e9
    best_images = None
    best_noise = None
    best_reward_dict: Optional[Dict[str, float]] = None

    reward_history: Dict[str, List[float]] = {"reward": [], "clip": [], "aesthetic": []}

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        eps_theta = adapter(base_noise)
        drift_reg = F.mse_loss(eps_theta, base_noise)

        if step % print_eigs_every == 0:
            with torch.no_grad():
                summarize_noise(eps_theta, "eps_theta")
                cur_eigs = covariance_eigenvalues_from_noise(eps_theta, patch_size=patch_size)
                print_eig_report(step=step, base_eigs=base_eigs, cur_eigs=cur_eigs)
                print_q_spectrum(step, adapter.orthogonal_matrix())

        images, _ = model(eps_theta, cond)
        reward_dict = reward_model(images)
        reward = reward_dict["reward"].mean()
        loss = -reward

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}: "
                f"loss={loss.item()}, reward={reward.item()}, reg={drift_reg.item()}"
            )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            current_reward = reward.item()
            current_clip = reward_dict["clip"].mean().item()
            current_aes = reward_dict["aesthetic"].mean().item()
            reward_history["reward"].append(current_reward)
            reward_history["clip"].append(current_clip)
            reward_history["aesthetic"].append(current_aes)
            # print(
            #     f"step={step:03d} | loss={loss.item():.4f} | reward={current_reward:.4f} "
            #     f"| clip={current_clip:.4f} | aes={current_aes:.4f} | reg={drift_reg.item():.6f}"
            # )

            if current_reward > best_reward:
                best_reward = current_reward
                best_images = images.detach().cpu()
                best_noise = eps_theta.detach().cpu()
                best_reward_dict = {
                    "reward": current_reward,
                    "clip": current_clip,
                    "aesthetic": current_aes,
                }
                if output_dir and save_outputs:
                    save_grid(best_images, os.path.join(output_dir, "best.png"))
                    torch.save(best_noise, os.path.join(output_dir, "best_noise.pt"))
            if save_outputs and output_dir and (step % save_every == 0 or step == steps):
                save_grid(images, os.path.join(output_dir, f"step_{step:03d}.png"))
                torch.save(adapter.state_dict(), os.path.join(output_dir, f"adapter_step_{step:03d}.pt"))

    if reward_history["reward"] and output_dir and save_outputs:
        steps_axis = list(range(1, len(reward_history["reward"]) + 1))
        reward_keys = list(reward_history.keys())
        num_subplots = len(reward_keys)
        fig, axs = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 4))
        if num_subplots == 1:
            axs = [axs]
        for idx, k in enumerate(reward_keys):
            v = reward_history[k]
            if all(not math.isfinite(x) for x in v):
                continue
            axs[idx].plot(steps_axis, v)
            axs[idx].set_xlabel("step")
            axs[idx].set_ylabel("score")
            axs[idx].set_title(f"{k} over steps")
            axs[idx].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "reward_history.png"))
        plt.close()

    if best_images is not None and output_dir and save_outputs:
        save_grid(best_images, os.path.join(output_dir, "final_best.png"))
    if output_dir and save_outputs:
        torch.save(adapter.state_dict(), os.path.join(output_dir, "adapter_final.pt"))

    # 返回图像列表（每张图 [C,H,W]）和最佳步的 reward 标量字典
    image_list: List[torch.Tensor] = []
    if best_images is not None:
        image_list = [best_images[i] for i in range(best_images.shape[0])]
    best_reward_value = float(best_reward_dict["reward"]) if best_reward_dict is not None else float(best_reward)
    best_clip_value = float(best_reward_dict["clip"]) if best_reward_dict is not None else float("nan")
    best_aes_value = float(best_reward_dict["aesthetic"]) if best_reward_dict is not None else float("nan")
    last_reward_value = float(reward_history["reward"][-1]) if reward_history["reward"] else float("nan")
    last_clip_value = float(reward_history["clip"][-1]) if reward_history["clip"] else float("nan")
    last_aes_value = float(reward_history["aesthetic"][-1]) if reward_history["aesthetic"] else float("nan")

    reward_out: Dict[str, object] = {
        # Backward-compatible keys: keep old behavior (best values).
        "reward": best_reward_value,
        "clip": best_clip_value,
        "aesthetic": best_aes_value,
        "best_reward": best_reward_value,
        "best_clip": best_clip_value,
        "best_aesthetic": best_aes_value,
        "last_reward": last_reward_value,
        "last_clip": last_clip_value,
        "last_aesthetic": last_aes_value,
        "reward_history": [float(x) for x in reward_history["reward"]],
        "clip_history": [float(x) for x in reward_history["clip"]],
        "aesthetic_history": [float(x) for x in reward_history["aesthetic"]],
    }
    # print(f"Done. Best reward = {best_reward:.4f}")
    # if output_dir and save_outputs:
    #     print(f"Saved to: {output_dir}")
    return image_list, reward_out


def run_test_time_oft_for_prompts(
    prompts: List[str],
    *,
    model_id: str = "/home/jiangzhou/.cache/modelscope/hub/models/AI-ModelScope/sdxl-turbo/",
    output_dir: Optional[str] = "outputs/test_time_oft_noise",
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
    steps: int = 30,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    noise_l2_weight: float = 1e-3,
    patch_size: int = 2,
    reward_type: str = "clip",
    aesthetic_ckpt: Optional[str] = None,
    clip_local_dir: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 42,
    variant: str = "fp16",
    save_every: int = 1,
    print_eigs_every: int = 1,
    save_outputs: bool = True,
) -> Dict[str, Tuple[List[torch.Tensor], Dict[str, object]]]:
    """
    依次对多个 prompt 运行 test-time OFT 优化。

    为简单起见，这里直接循环调用 `run_test_time_oft`，
    方便你在外部只写一段循环代码。

    Returns:
        一个字典：key 为 prompt 字符串，value 为
        `(image_list, reward_dict)`。
    """
    results: Dict[str, Tuple[List[torch.Tensor], Dict[str, object]]] = {}

    for i, p in tqdm(enumerate(prompts), total=len(prompts), desc="Running test-time OFT"):
        cur_output_dir = output_dir
        if output_dir is not None and save_outputs:
            # 区分不同 prompt 的输出目录
            cur_output_dir = os.path.join(output_dir, f"prompt_{i}")

        images, rewards = run_test_time_oft(
            p,
            model_id=model_id,
            output_dir=cur_output_dir,
            height=height,
            width=width,
            batch_size=batch_size,
            steps=steps,
            lr=lr,
            weight_decay=weight_decay,
            noise_l2_weight=noise_l2_weight,
            patch_size=patch_size,
            reward_type=reward_type,
            aesthetic_ckpt=aesthetic_ckpt,
            clip_local_dir=clip_local_dir,
            device=device,
            seed=seed,
            variant=variant,
            save_every=save_every,
            print_eigs_every=print_eigs_every,
            save_outputs=save_outputs,
        )
        results[p] = (images, rewards)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--model_id",
        type=str,
        default="/home/jiangzhou/.cache/modelscope/hub/models/AI-ModelScope/sdxl-turbo/",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/test_time_oft_noise")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--noise_l2_weight", type=float, default=1e-3)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--reward_type", type=str, default="clip", choices=["clip", "aesthetic", "hybrid"])
    parser.add_argument("--aesthetic_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", type=str, default="fp16")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--print_eigs_every", type=int, default=1)
    parser.add_argument(
        "--clip_local_dir",
        type=str,
        default=None,
        help="Local Hugging Face CLIP snapshot dir, e.g. /home/.../models--openai--clip-vit-large-patch14/snapshots/xxx",
    )
    args = parser.parse_args()

    image_list, reward_dict = run_test_time_oft(
        args.prompt,
        model_id=args.model_id,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        noise_l2_weight=args.noise_l2_weight,
        patch_size=args.patch_size,
        reward_type=args.reward_type,
        aesthetic_ckpt=args.aesthetic_ckpt,
        clip_local_dir=args.clip_local_dir,
        device=args.device,
        seed=args.seed,
        variant=args.variant,
        save_every=args.save_every,
        print_eigs_every=args.print_eigs_every,
        save_outputs=True,
    )
    return image_list, reward_dict


if __name__ == "__main__":
    main()