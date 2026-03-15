import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms

from diffusers import DDPMScheduler, StableDiffusionPipeline


# -------------------------------------------------
# Example
'''
python test_time_oft_noise_sd15_sft.py \
  --prompt "a cinematic portrait of a girl in soft light, highly detailed" \
  --target_image_path ./target.jpg \
  --steps 30 \
  --lr 5e-4 \
  --patch_size 2 \
  --output_dir outputs/run_sft_sd15 \
  --model_id ./models/Hyper-SD15-1step
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

_GLOBAL_PIPE: Optional[StableDiffusionPipeline] = None
_GLOBAL_PIPE_KEY: Optional[Tuple[str, str, str]] = None


# -----------------------------
# Patch-wise orthogonal epsilon transform
# -----------------------------

class PatchOrthogonalNoise(nn.Module):
    r"""
    Shared orthogonal transform applied on flattened latent patches.
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
        # q = torch.matrix_exp(skew)
        
        # Cayley transform:
        # Q = (I - A)^(-1) (I + A)
        # use solve instead of explicit inverse for better stability
        eye = torch.eye(self.block_dim, device=a.device, dtype=a.dtype)
        q = torch.linalg.solve(eye - a, eye + a)
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
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(b, -1, self.block_dim)
        x = x @ q.T
        x = x.view(b, h // p, w // p, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, h, w)
        return x


# -----------------------------
# Noise spectrum diagnostics
# -----------------------------

def noise_to_patch_matrix(noise: torch.Tensor, patch_size: int) -> torch.Tensor:
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


def print_q_spectrum(step: int, q: torch.Tensor) -> None:
    svals = torch.linalg.svdvals(q.detach().float().cpu())
    # print(
    #     f"[step {step:03d}] Q singular values min/max = "
    #     f"{svals.min().item():.8f} / {svals.max().item():.8f}"
    # )


# -----------------------------
# Prompt condition
# -----------------------------

@dataclass
class PromptCond:
    encoder_hidden_states: torch.Tensor


# -----------------------------
# SD1.5 differentiable one-step forward
# -----------------------------

class OneStepSD15(nn.Module):
    def __init__(self, pipe: StableDiffusionPipeline, height: int = 512, width: int = 512):
        super().__init__()
        self.pipe = pipe
        self.height = height
        self.width = width
        self.vae_scale_factor = pipe.vae_scale_factor
        self.latent_h = height // self.vae_scale_factor
        self.latent_w = width // self.vae_scale_factor

    @property
    def device(self):
        return next(self.pipe.unet.parameters()).device

    @property
    def model_dtype(self):
        return self.pipe.unet.dtype

    def build_prompt_cond(self, prompt: str, device: str, batch_size: int = 1) -> PromptCond:
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder

        text_inputs = tokenizer(
            [prompt] * batch_size,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        encoder_hidden_states = text_encoder(input_ids)[0]

        return PromptCond(
            encoder_hidden_states=encoder_hidden_states,
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

    def encode_image_to_latent(self, images_01: torch.Tensor) -> torch.Tensor:
        """
        images_01: [B, 3, H, W] in [0, 1]
        return: latent in SD1.5 latent space
        """
        vae = self.pipe.vae
        x = images_01.to(device=vae.device, dtype=torch.float32)
        x = x * 2.0 - 1.0

        if x.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                posterior = vae.encode(x).latent_dist
                latents = posterior.mean
        else:
            posterior = vae.encode(x).latent_dist
            latents = posterior.mean

        latents = latents * vae.config.scaling_factor
        return latents

    def forward(self, latents: torch.Tensor, cond: PromptCond) -> Tuple[torch.Tensor, torch.Tensor]:
        scheduler = self.pipe.scheduler

        # one-step inference
        scheduler.set_timesteps(1, device=latents.device)
        t = scheduler.timesteps[0]

        latent_model_input = scheduler.scale_model_input(latents.to(self.model_dtype), t)

        noise_pred = self.pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=cond.encoder_hidden_states,
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
# Target image loader
# -----------------------------

def load_target_image(
    image_path: str,
    height: int,
    width: int,
    device: str,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    image = tfm(image).unsqueeze(0).to(device)
    return image


# -----------------------------
# Main optimization loop
# -----------------------------

def save_grid(images: torch.Tensor, path: str) -> None:
    nrow = max(1, int(math.sqrt(images.shape[0])))
    save_image(images.detach().cpu(), path, nrow=nrow)


def run_test_time_oft_sft(
    prompt: str,
    *,
    target_image_path: str,
    model_id: str = "./models/stable-diffusion-v1-5",
    output_dir: Optional[str] = "outputs/test_time_oft_sft_sd15",
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
    steps: int = 30,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    patch_size: int = 2,
    latent_loss_weight: float = 1.0,
    pixel_l1_weight: float = 0.0,
    drift_reg_weight: float = 0.0,
    device: Optional[str] = None,
    seed: int = 42,
    variant: Optional[str] = None,
    save_every: int = 1,
    print_eigs_every: int = 1,
    save_outputs: bool = True,
) -> Tuple[List[torch.Tensor], Dict[str, object]]:
    """
    用目标图的监督损失替代 reward：
    - latent MSE between denoised latent and target latent
    - optional pixel L1 between decoded image and target image

    注意：
    vanilla SD1.5 不是 one-step 蒸馏模型，这里只是把原先 SDXL Turbo 的结构
    等价改写到 SD1.5 上，代码可跑，但一跳生成质量通常不会特别好。
    """
    if output_dir and save_outputs:
        ensure_dir(output_dir)

    set_seed(seed)
    dev = get_device(device)
    torch.set_grad_enabled(True)

    global _GLOBAL_PIPE, _GLOBAL_PIPE_KEY
    pipe_key = (model_id, dev, str(variant))

    if _GLOBAL_PIPE is None or _GLOBAL_PIPE_KEY != pipe_key:
        load_kwargs = {
            "torch_dtype": torch.float16 if dev.startswith("cuda") else torch.float32,
            "safety_checker": None,
        }
        if dev.startswith("cuda") and variant is not None:
            load_kwargs["variant"] = variant

        try:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, **load_kwargs)
        except Exception:
            load_kwargs.pop("variant", None)
            pipe = StableDiffusionPipeline.from_pretrained(model_id, **load_kwargs)

        pipe = pipe.to(dev)

        # 对 SD1.5，显式换成 DDPMScheduler，和训练形式更一致
        pipe.scheduler = DDPMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
        )

        freeze_module(pipe.unet)
        freeze_module(pipe.vae)
        if getattr(pipe, "text_encoder", None) is not None:
            freeze_module(pipe.text_encoder)

        # VAE decode / encode 维持 fp32，稳定一些
        pipe.vae.to(dtype=torch.float32)

        _GLOBAL_PIPE = pipe
        _GLOBAL_PIPE_KEY = pipe_key

    pipe = _GLOBAL_PIPE
    model = OneStepSD15(pipe, height=height, width=width)
    cond = model.build_prompt_cond(prompt, dev, batch_size=batch_size)

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

    target_image = load_target_image(target_image_path, height, width, dev)
    with torch.no_grad():
        target_latent = model.encode_image_to_latent(target_image)

    if batch_size > 1:
        target_image = target_image.repeat(batch_size, 1, 1, 1)
        target_latent = target_latent.repeat(batch_size, 1, 1, 1)

    if output_dir and save_outputs:
        save_grid(target_image, os.path.join(output_dir, "target.png"))

    with torch.no_grad():
        init_images, init_denoised = model(base_noise.to(dev), cond)
        if output_dir and save_outputs:
            save_grid(init_images, os.path.join(output_dir, "init.png"))

    best_loss = float("inf")
    best_images = None
    best_noise = None
    best_stats: Optional[Dict[str, float]] = None

    loss_history: Dict[str, List[float]] = {
        "total_loss": [],
        "latent_loss": [],
        "pixel_l1": [],
        "drift_reg": [],
    }

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

        images, denoised = model(eps_theta, cond)

        latent_loss = F.mse_loss(denoised, target_latent)
        pixel_l1 = F.l1_loss(images, target_image) if pixel_l1_weight > 0 else torch.zeros_like(latent_loss)

        loss = (
            latent_loss_weight * latent_loss
            + pixel_l1_weight * pixel_l1
            + drift_reg_weight * drift_reg
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}: "
                f"loss={loss.item()}, latent_loss={latent_loss.item()}, "
                f"pixel_l1={pixel_l1.item()}, drift_reg={drift_reg.item()}"
            )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss_val = float(loss.item())
            latent_loss_val = float(latent_loss.item())
            pixel_l1_val = float(pixel_l1.item())
            drift_reg_val = float(drift_reg.item())

            loss_history["total_loss"].append(total_loss_val)
            loss_history["latent_loss"].append(latent_loss_val)
            loss_history["pixel_l1"].append(pixel_l1_val)
            loss_history["drift_reg"].append(drift_reg_val)

            if total_loss_val < best_loss:
                best_loss = total_loss_val
                best_images = images.detach().cpu()
                best_noise = eps_theta.detach().cpu()
                best_stats = {
                    "total_loss": total_loss_val,
                    "latent_loss": latent_loss_val,
                    "pixel_l1": pixel_l1_val,
                    "drift_reg": drift_reg_val,
                }
                if output_dir and save_outputs:
                    save_grid(best_images, os.path.join(output_dir, "best.png"))
                    torch.save(best_noise, os.path.join(output_dir, "best_noise.pt"))

            if save_outputs and output_dir and (step % save_every == 0 or step == steps):
                save_grid(images, os.path.join(output_dir, f"step_{step:03d}.png"))
                torch.save(adapter.state_dict(), os.path.join(output_dir, f"adapter_step_{step:03d}.pt"))

    if loss_history["total_loss"] and output_dir and save_outputs:
        steps_axis = list(range(1, len(loss_history["total_loss"]) + 1))
        loss_keys = list(loss_history.keys())
        num_subplots = len(loss_keys)
        fig, axs = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 4))
        if num_subplots == 1:
            axs = [axs]
        for idx, k in enumerate(loss_keys):
            v = loss_history[k]
            axs[idx].plot(steps_axis, v)
            axs[idx].set_xlabel("step")
            axs[idx].set_ylabel("loss")
            axs[idx].set_title(f"{k} over steps")
            axs[idx].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_history.png"))
        plt.close()

    if best_images is not None and output_dir and save_outputs:
        save_grid(best_images, os.path.join(output_dir, "final_best.png"))
    if output_dir and save_outputs:
        torch.save(adapter.state_dict(), os.path.join(output_dir, "adapter_final.pt"))

    image_list: List[torch.Tensor] = []
    if best_images is not None:
        image_list = [best_images[i] for i in range(best_images.shape[0])]

    best_total_loss = float(best_stats["total_loss"]) if best_stats is not None else float(best_loss)
    best_latent_loss = float(best_stats["latent_loss"]) if best_stats is not None else float("nan")
    best_pixel_l1 = float(best_stats["pixel_l1"]) if best_stats is not None else float("nan")
    best_drift_reg = float(best_stats["drift_reg"]) if best_stats is not None else float("nan")

    last_total_loss = float(loss_history["total_loss"][-1]) if loss_history["total_loss"] else float("nan")
    last_latent_loss = float(loss_history["latent_loss"][-1]) if loss_history["latent_loss"] else float("nan")
    last_pixel_l1 = float(loss_history["pixel_l1"][-1]) if loss_history["pixel_l1"] else float("nan")
    last_drift_reg = float(loss_history["drift_reg"][-1]) if loss_history["drift_reg"] else float("nan")

    stats_out: Dict[str, object] = {
        "loss": best_total_loss,
        "latent_loss": best_latent_loss,
        "pixel_l1": best_pixel_l1,
        "drift_reg": best_drift_reg,
        "best_loss": best_total_loss,
        "best_latent_loss": best_latent_loss,
        "best_pixel_l1": best_pixel_l1,
        "best_drift_reg": best_drift_reg,
        "last_loss": last_total_loss,
        "last_latent_loss": last_latent_loss,
        "last_pixel_l1": last_pixel_l1,
        "last_drift_reg": last_drift_reg,
        "loss_history": [float(x) for x in loss_history["total_loss"]],
        "latent_loss_history": [float(x) for x in loss_history["latent_loss"]],
        "pixel_l1_history": [float(x) for x in loss_history["pixel_l1"]],
        "drift_reg_history": [float(x) for x in loss_history["drift_reg"]],
    }

    return image_list, stats_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--target_image_path", type=str, required=True)
    parser.add_argument(
        "--model_id",
        type=str,
        default="./models/stable-diffusion-v1-5",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/test_time_oft_sft_sd15")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--latent_loss_weight", type=float, default=1.0)
    parser.add_argument("--pixel_l1_weight", type=float, default=0.0)
    parser.add_argument("--drift_reg_weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--print_eigs_every", type=int, default=1)
    args = parser.parse_args()

    image_list, stats_dict = run_test_time_oft_sft(
        args.prompt,
        target_image_path=args.target_image_path,
        model_id=args.model_id,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patch_size=args.patch_size,
        latent_loss_weight=args.latent_loss_weight,
        pixel_l1_weight=args.pixel_l1_weight,
        drift_reg_weight=args.drift_reg_weight,
        device=args.device,
        seed=args.seed,
        variant=args.variant,
        save_every=args.save_every,
        print_eigs_every=args.print_eigs_every,
        save_outputs=True,
    )
    return image_list, stats_dict


if __name__ == "__main__":
    main()