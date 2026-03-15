import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from transformers import AutoModel, AutoProcessor

from test_time_oft_noise_sdxl_turbo_sdxlpipe_full import (
    OneStepSDXLTurbo,
    PatchOrthogonalNoise,
    covariance_eigenvalues_from_noise,
    ensure_dir,
    freeze_module,
    get_device,
    print_eig_report,
    print_q_spectrum,
    save_grid,
    set_seed,
    summarize_noise,
)


_GLOBAL_PIPE: Optional[StableDiffusionXLPipeline] = None
_GLOBAL_PIPE_KEY: Optional[Tuple[str, str, str]] = None
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
        self.pickscore_model_id = pickscore_model_id

        global _GLOBAL_PICKSCORE_MODEL, _GLOBAL_PICKSCORE_PROCESSOR, _GLOBAL_PICKSCORE_ID
        if _GLOBAL_PICKSCORE_MODEL is None or _GLOBAL_PICKSCORE_ID != pickscore_model_id:
            _GLOBAL_PICKSCORE_MODEL = AutoModel.from_pretrained(pickscore_model_id).eval()
            _GLOBAL_PICKSCORE_PROCESSOR = AutoProcessor.from_pretrained(pickscore_model_id)
            _GLOBAL_PICKSCORE_ID = pickscore_model_id

        self.model = _GLOBAL_PICKSCORE_MODEL
        self.processor = _GLOBAL_PICKSCORE_PROCESSOR
        freeze_module(self.model)

        image_processor = self.processor.image_processor
        size_cfg = image_processor.size
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
        embed_dim = getattr(self.model.config, "projection_dim", 768)
        self.register_buffer("text_features", torch.zeros(1, embed_dim), persistent=False)

    @staticmethod
    def _unwrap_features(feats) -> torch.Tensor:
        if isinstance(feats, torch.Tensor):
            return feats
        # Compatibility across transformers versions/models.
        if hasattr(feats, "text_embeds") and feats.text_embeds is not None:
            return feats.text_embeds
        if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            return feats.pooler_output
        if hasattr(feats, "last_hidden_state") and feats.last_hidden_state is not None:
            return feats.last_hidden_state[:, 0]
        if isinstance(feats, (tuple, list)) and len(feats) > 0 and isinstance(feats[0], torch.Tensor):
            return feats[0]
        raise TypeError(f"Unsupported text feature output type: {type(feats)}")

    def set_device(self, device: str):
        self.to(device)
        with torch.no_grad():
            txt = self.model.get_text_features(
                input_ids=self.text_input_ids.to(device),
                attention_mask=self.text_attention_mask.to(device),
            )
            txt = self._unwrap_features(txt)
            txt = F.normalize(txt.float(), dim=-1)
            if txt.shape[-1] != self.text_features.shape[-1]:
                self.text_features = txt.detach().clone()
            else:
                self.text_features.copy_(txt)
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
        image_features = self._unwrap_features(image_features)
        image_features = F.normalize(image_features.float(), dim=-1)
        pickscore = (image_features * self.text_features).sum(dim=-1)
        return {
            "reward": pickscore,
            "pickscore": pickscore,
        }


def run_test_time_oft_pickscore(
    prompt: str,
    *,
    model_id: str = "/home/jiangzhou/.cache/modelscope/hub/models/AI-ModelScope/sdxl-turbo/",
    pickscore_model_id: str = "yuvalkirstain/PickScore_v1",
    output_dir: Optional[str] = "outputs/test_time_oft_pickscore",
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
    steps: int = 30,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    patch_size: int = 2,
    device: Optional[str] = None,
    seed: int = 42,
    variant: str = "fp16",
    save_every: int = 1,
    print_eigs_every: int = 1,
    save_outputs: bool = True,
) -> Tuple[List[torch.Tensor], Dict[str, object]]:
    if output_dir and save_outputs:
        ensure_dir(output_dir)
    set_seed(seed)
    dev = get_device(device)
    torch.set_grad_enabled(True)

    global _GLOBAL_PIPE, _GLOBAL_PIPE_KEY
    pipe_key = (model_id, dev, variant if dev.startswith("cuda") else "none")
    if _GLOBAL_PIPE is None or _GLOBAL_PIPE_KEY != pipe_key:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if dev.startswith("cuda") else torch.float32,
            variant=variant if dev.startswith("cuda") else None,
        ).to(dev)
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
    pipe = _GLOBAL_PIPE

    model = OneStepSDXLTurbo(pipe, height=height, width=width)
    cond = model.build_prompt_cond(prompt, dev, batch_size=batch_size)
    reward_model = PickScoreReward(prompt=prompt, pickscore_model_id=pickscore_model_id).set_device(dev)

    adapter = PatchOrthogonalNoise(
        channels=pipe.unet.config.in_channels,
        patch_size=patch_size,
    ).to(dev)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)

    gen_device = dev if dev.startswith("cuda") else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    base_noise = model.sample_base_noise(batch_size, generator=generator)
    base_eigs = covariance_eigenvalues_from_noise(base_noise, patch_size=patch_size)
    summarize_noise(base_noise, "base_noise")

    with torch.no_grad():
        init_images, _ = model(base_noise.to(dev), cond)
        if output_dir and save_outputs:
            save_grid(init_images, os.path.join(output_dir, "init.png"))

    best_reward = -1e9
    best_images = None
    best_reward_dict: Optional[Dict[str, float]] = None
    reward_history: Dict[str, List[float]] = {"reward": [], "pickscore": []}

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        eps_theta = adapter(base_noise)

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
            raise RuntimeError(f"Non-finite loss at step {step}: loss={loss.item()}, reward={reward.item()}")

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            current_reward = reward.item()
            current_pick = reward_dict["pickscore"].mean().item()
            reward_history["reward"].append(current_reward)
            reward_history["pickscore"].append(current_pick)
            if current_reward > best_reward:
                best_reward = current_reward
                best_images = images.detach().cpu()
                best_reward_dict = {"reward": current_reward, "pickscore": current_pick}
                if output_dir and save_outputs:
                    save_grid(best_images, os.path.join(output_dir, "best.png"))
            if output_dir and save_outputs and (step % save_every == 0 or step == steps):
                save_grid(images, os.path.join(output_dir, f"step_{step:03d}.png"))

    if reward_history["reward"] and output_dir and save_outputs:
        steps_axis = list(range(1, len(reward_history["reward"]) + 1))
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for idx, k in enumerate(["reward", "pickscore"]):
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

    image_list: List[torch.Tensor] = []
    if best_images is not None:
        image_list = [best_images[i] for i in range(best_images.shape[0])]

    best_reward_value = float(best_reward_dict["reward"]) if best_reward_dict is not None else float(best_reward)
    best_pick_value = float(best_reward_dict["pickscore"]) if best_reward_dict is not None else float("nan")
    last_reward_value = float(reward_history["reward"][-1]) if reward_history["reward"] else float("nan")
    last_pick_value = float(reward_history["pickscore"][-1]) if reward_history["pickscore"] else float("nan")

    reward_out: Dict[str, object] = {
        # Backward-compatible keys: keep using best metrics here.
        "reward": best_reward_value,
        "pickscore": best_pick_value,
        "best_reward": best_reward_value,
        "best_pickscore": best_pick_value,
        "last_reward": last_reward_value,
        "last_pickscore": last_pick_value,
        "reward_history": [float(x) for x in reward_history["reward"]],
        "pickscore_history": [float(x) for x in reward_history["pickscore"]],
    }
    return image_list, reward_out


def run_test_time_oft_pickscore_for_prompts(
    prompts: List[str],
    **kwargs,
) -> Dict[str, Tuple[List[torch.Tensor], Dict[str, object]]]:
    results: Dict[str, Tuple[List[torch.Tensor], Dict[str, object]]] = {}
    for i, p in enumerate(prompts):
        cur_output_dir = kwargs.get("output_dir")
        if cur_output_dir is not None and kwargs.get("save_outputs", True):
            cur_output_dir = os.path.join(cur_output_dir, f"prompt_{i}")
        local_kwargs = dict(kwargs)
        local_kwargs["output_dir"] = cur_output_dir
        images, rewards = run_test_time_oft_pickscore(p, **local_kwargs)
        results[p] = (images, rewards)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="/home/jiangzhou/.cache/modelscope/hub/models/AI-ModelScope/sdxl-turbo/")
    parser.add_argument("--pickscore_model_id", type=str, default="yuvalkirstain/PickScore_v1")
    parser.add_argument("--output_dir", type=str, default="outputs/test_time_oft_pickscore")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", type=str, default="fp16")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--print_eigs_every", type=int, default=1)
    args = parser.parse_args()

    run_test_time_oft_pickscore(
        args.prompt,
        model_id=args.model_id,
        pickscore_model_id=args.pickscore_model_id,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patch_size=args.patch_size,
        device=args.device,
        seed=args.seed,
        variant=args.variant,
        save_every=args.save_every,
        print_eigs_every=args.print_eigs_every,
        save_outputs=True,
    )


if __name__ == "__main__":
    main()
