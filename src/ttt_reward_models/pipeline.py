from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline

from .utils import freeze_module


_GLOBAL_PIPE: Optional[StableDiffusionXLPipeline] = None
_GLOBAL_PIPE_KEY: Optional[Tuple[str, str, str]] = None


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

    def encode_image_to_latent(self, images_01: torch.Tensor) -> torch.Tensor:
        """
        Encode images in [0, 1] into the SDXL latent space using posterior mean.
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

    def forward(self, latents: torch.Tensor, cond: PromptCond):
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



def get_cached_sdxl_pipeline(model_id: str, device: str, variant: str = "fp16") -> StableDiffusionXLPipeline:
    global _GLOBAL_PIPE, _GLOBAL_PIPE_KEY
    pipe_key = (model_id, device, variant if device.startswith("cuda") else "none")

    if _GLOBAL_PIPE is None or _GLOBAL_PIPE_KEY != pipe_key:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            variant=variant if device.startswith("cuda") else None,
        )
        pipe = pipe.to(device)
        pipe.scheduler = pipe.scheduler.__class__.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
        )

        freeze_module(pipe.unet)
        freeze_module(pipe.vae)
        if getattr(pipe, "text_encoder", None) is not None:
            freeze_module(pipe.text_encoder)
        if getattr(pipe, "text_encoder_2", None) is not None:
            freeze_module(pipe.text_encoder_2)
        pipe.vae.to(dtype=torch.float32)

        _GLOBAL_PIPE = pipe
        _GLOBAL_PIPE_KEY = pipe_key

    return _GLOBAL_PIPE
