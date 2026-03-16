import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .adapters import PatchOrthogonalNoise
from .diagnostics import (
    covariance_eigenvalues_from_noise,
    print_eig_report,
    print_q_spectrum,
    save_grid,
    save_noise_distribution_report,
    summarize_noise,
)
from .pipeline import OneStepSDXLTurbo, get_cached_sdxl_pipeline
from .rewards_clip import CLIPReward
from .rewards_hpsv2 import HPSv2Reward
from .rewards_imagereward import ImageRewardReward
from .rewards_pickscore import PickScoreReward
from .utils import ensure_dir, get_device, set_seed


def _run_test_time_oft_core(
    prompt: str,
    *,
    reward_model,
    metric_keys: List[str],
    model_id: str = 'models/sdxl-turbo',
    output_dir: Optional[str] = 'outputs/test_time_oft_noise',
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
    steps: int = 30,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    noise_l2_weight: float = 1e-3,
    patch_size: int = 2,
    device: Optional[str] = None,
    seed: int = 42,
    variant: str = 'fp16',
    save_every: int = 1,
    print_eigs_every: int = 1,
    save_outputs: bool = True,
) -> Tuple[List[torch.Tensor], Dict[str, object]]:
    if output_dir and save_outputs:
        ensure_dir(output_dir)
    set_seed(seed)
    dev = get_device(device)
    torch.set_grad_enabled(True)

    pipe = get_cached_sdxl_pipeline(model_id=model_id, device=dev, variant=variant)
    model = OneStepSDXLTurbo(pipe, height=height, width=width)
    cond = model.build_prompt_cond(prompt, dev, batch_size=batch_size)
    reward_model = reward_model.set_device(dev)

    adapter = PatchOrthogonalNoise(
        channels=pipe.unet.config.in_channels,
        patch_size=patch_size,
    ).to(dev)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)

    gen_device = dev if dev.startswith('cuda') else 'cpu'
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)
    base_noise = model.sample_base_noise(batch_size, generator=generator)

    base_eigs = covariance_eigenvalues_from_noise(base_noise, patch_size=patch_size)
    summarize_noise(base_noise, 'base_noise')

    if output_dir and save_outputs:
        with torch.no_grad():
            init_eps_theta = adapter(base_noise)
            save_noise_distribution_report(
                base_noise=base_noise,
                transformed_noise=init_eps_theta,
                output_dir=output_dir,
                patch_size=patch_size,
                prefix='orthogonal_gaussian_init',
                title='Before optimization: base noise vs orthogonally transformed noise',
                label_a='base Gaussian noise',
                label_b='Q @ base noise (init)',
            )

    with torch.no_grad():
        init_images, _ = model(base_noise.to(dev), cond)
        if output_dir and save_outputs:
            save_grid(init_images, os.path.join(output_dir, 'init.png'))

    best_reward = -1e9
    best_images = None
    best_noise = None
    best_reward_dict: Optional[Dict[str, float]] = None
    history_keys = ['reward'] + metric_keys
    reward_history: Dict[str, List[float]] = {k: [] for k in history_keys}

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        eps_theta = adapter(base_noise)
        drift_reg = F.mse_loss(eps_theta, base_noise)

        if step % print_eigs_every == 0:
            with torch.no_grad():
                summarize_noise(eps_theta, 'eps_theta')
                cur_eigs = covariance_eigenvalues_from_noise(eps_theta, patch_size=patch_size)
                print_eig_report(step=step, base_eigs=base_eigs, cur_eigs=cur_eigs)
                print_q_spectrum(step, adapter.orthogonal_matrix())

        images, _ = model(eps_theta, cond)
        reward_dict = reward_model(images)
        reward = reward_dict['reward'].mean()

        loss = -reward + noise_l2_weight * drift_reg

        if not torch.isfinite(loss):
            raise RuntimeError(
                f'Non-finite loss at step {step}: '
                f'loss={loss.item()}, reward={reward.item()}, reg={drift_reg.item()}'
            )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            current_reward = float(reward.item())
            reward_history['reward'].append(current_reward)
            current_record = {'reward': current_reward}
            for k in metric_keys:
                val = reward_dict[k].mean().item() if k in reward_dict else float('nan')
                reward_history[k].append(float(val))
                current_record[k] = float(val)

            if current_reward > best_reward:
                best_reward = current_reward
                best_images = images.detach().cpu()
                best_noise = eps_theta.detach().cpu()
                best_reward_dict = current_record
                if output_dir and save_outputs:
                    save_grid(best_images, os.path.join(output_dir, 'best.png'))
                    torch.save(best_noise, os.path.join(output_dir, 'best_noise.pt'))
            if save_outputs and output_dir and (step % save_every == 0 or step == steps):
                save_grid(images, os.path.join(output_dir, f'step_{step:03d}.png'))
                torch.save(adapter.state_dict(), os.path.join(output_dir, f'adapter_step_{step:03d}.pt'))

    if reward_history['reward'] and output_dir and save_outputs:
        steps_axis = list(range(1, len(reward_history['reward']) + 1))
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
            axs[idx].set_xlabel('step')
            axs[idx].set_ylabel('score')
            axs[idx].set_title(f'{k} over steps')
            axs[idx].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reward_history.png'))
        plt.close()

    if best_images is not None and output_dir and save_outputs:
        save_grid(best_images, os.path.join(output_dir, 'final_best.png'))
    if output_dir and save_outputs:
        torch.save(adapter.state_dict(), os.path.join(output_dir, 'adapter_final.pt'))
        with torch.no_grad():
            final_noise = best_noise if best_noise is not None else adapter(base_noise).detach().cpu()
            save_noise_distribution_report(
                base_noise=base_noise.detach().cpu(),
                transformed_noise=final_noise,
                output_dir=output_dir,
                patch_size=patch_size,
                prefix='orthogonal_gaussian_final',
                title='After optimization: base noise vs orthogonally transformed noise',
                label_a='base Gaussian noise',
                label_b='Q @ base noise (final/best)',
            )

    image_list: List[torch.Tensor] = []
    if best_images is not None:
        image_list = [best_images[i] for i in range(best_images.shape[0])]

    reward_out: Dict[str, object] = {
        'reward': float(best_reward_dict['reward']) if best_reward_dict is not None else float(best_reward),
        'best_reward': float(best_reward_dict['reward']) if best_reward_dict is not None else float(best_reward),
        'last_reward': float(reward_history['reward'][-1]) if reward_history['reward'] else float('nan'),
        'reward_history': [float(x) for x in reward_history['reward']],
    }
    for k in metric_keys:
        best_val = float(best_reward_dict[k]) if best_reward_dict is not None and k in best_reward_dict else float('nan')
        last_val = float(reward_history[k][-1]) if reward_history.get(k) else float('nan')
        reward_out[k] = best_val
        reward_out[f'best_{k}'] = best_val
        reward_out[f'last_{k}'] = last_val
        reward_out[f'{k}_history'] = [float(x) for x in reward_history[k]]
    return image_list, reward_out


def run_test_time_oft(
    prompt: str,
    *,
    model_id: str = 'models/sdxl-turbo',
    output_dir: Optional[str] = 'outputs/test_time_oft_noise',
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
    steps: int = 30,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    noise_l2_weight: float = 1e-3,
    patch_size: int = 2,
    reward_type: str = 'clip',
    aesthetic_ckpt: Optional[str] = None,
    clip_local_dir: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 42,
    variant: str = 'fp16',
    save_every: int = 1,
    print_eigs_every: int = 1,
    save_outputs: bool = True,
) -> Tuple[List[torch.Tensor], Dict[str, object]]:
    reward_model = CLIPReward(
        prompt=prompt,
        reward_type=reward_type,
        aesthetic_ckpt=aesthetic_ckpt,
        clip_local_dir=clip_local_dir,
    )
    return _run_test_time_oft_core(
        prompt,
        reward_model=reward_model,
        metric_keys=['clip', 'aesthetic'],
        model_id=model_id,
        output_dir=output_dir,
        height=height,
        width=width,
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        weight_decay=weight_decay,
        noise_l2_weight=noise_l2_weight,
        patch_size=patch_size,
        device=device,
        seed=seed,
        variant=variant,
        save_every=save_every,
        print_eigs_every=print_eigs_every,
        save_outputs=save_outputs,
    )


def run_test_time_oft_for_prompts(
    prompts: List[str],
    **kwargs,
) -> Dict[str, Tuple[List[torch.Tensor], Dict[str, object]]]:
    results: Dict[str, Tuple[List[torch.Tensor], Dict[str, object]]] = {}
    for i, p in tqdm(enumerate(prompts), total=len(prompts), desc='Running test-time OFT'):
        cur_output_dir = kwargs.get('output_dir')
        if cur_output_dir is not None and kwargs.get('save_outputs', True):
            cur_output_dir = os.path.join(cur_output_dir, f'prompt_{i}')
        local_kwargs = dict(kwargs)
        local_kwargs['output_dir'] = cur_output_dir
        images, rewards = run_test_time_oft(p, **local_kwargs)
        results[p] = (images, rewards)
    return results


def run_test_time_oft_pickscore(
    prompt: str,
    *,
    model_id: str = 'models/sdxl-turbo',
    pickscore_model_id: str = 'yuvalkirstain/PickScore_v1',
    output_dir: Optional[str] = 'outputs/test_time_oft_pickscore',
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
    steps: int = 30,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    patch_size: int = 2,
    device: Optional[str] = None,
    seed: int = 42,
    variant: str = 'fp16',
    save_every: int = 1,
    print_eigs_every: int = 1,
    save_outputs: bool = True,
) -> Tuple[List[torch.Tensor], Dict[str, object]]:
    reward_model = PickScoreReward(prompt=prompt, pickscore_model_id=pickscore_model_id)
    return _run_test_time_oft_core(
        prompt,
        reward_model=reward_model,
        metric_keys=['pickscore'],
        model_id=model_id,
        output_dir=output_dir,
        height=height,
        width=width,
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        weight_decay=weight_decay,
        patch_size=patch_size,
        device=device,
        seed=seed,
        variant=variant,
        save_every=save_every,
        print_eigs_every=print_eigs_every,
        save_outputs=save_outputs,
    )


def run_test_time_oft_imagereward(
    prompt: str,
    *,
    model_id: str = 'models/sdxl-turbo',
    output_dir: Optional[str] = 'outputs/test_time_oft_imagereward',
    imagereward_root: Optional[str] = None,
    imagereward_model_path: Optional[str] = None,
    imagereward_med_config_path: Optional[str] = None,
    imagereward_auto_download: bool = False,
    prefer_modelscope: bool = True,
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
    steps: int = 30,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    patch_size: int = 2,
    device: Optional[str] = None,
    seed: int = 42,
    variant: str = 'fp16',
    save_every: int = 1,
    print_eigs_every: int = 1,
    save_outputs: bool = True,
) -> Tuple[List[torch.Tensor], Dict[str, object]]:
    reward_model = ImageRewardReward(
        prompt=prompt,
        imagereward_root=imagereward_root,
        imagereward_model_path=imagereward_model_path,
        imagereward_med_config_path=imagereward_med_config_path,
        auto_download=imagereward_auto_download,
        prefer_modelscope=prefer_modelscope,
    )
    return _run_test_time_oft_core(
        prompt,
        reward_model=reward_model,
        metric_keys=['imagereward'],
        model_id=model_id,
        output_dir=output_dir,
        height=height,
        width=width,
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        weight_decay=weight_decay,
        patch_size=patch_size,
        device=device,
        seed=seed,
        variant=variant,
        save_every=save_every,
        print_eigs_every=print_eigs_every,
        save_outputs=save_outputs,
    )


def run_test_time_oft_hpsv2(
    prompt: str,
    *,
    model_id: str = 'models/sdxl-turbo',
    output_dir: Optional[str] = 'outputs/test_time_oft_hpsv2',
    hps_version: str = 'v2.1',
    hps_root: Optional[str] = None,
    hps_checkpoint_path: Optional[str] = None,
    hps_auto_download: bool = False,
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
    steps: int = 30,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    patch_size: int = 2,
    device: Optional[str] = None,
    seed: int = 42,
    variant: str = 'fp16',
    save_every: int = 1,
    print_eigs_every: int = 1,
    save_outputs: bool = True,
) -> Tuple[List[torch.Tensor], Dict[str, object]]:
    reward_model = HPSv2Reward(
        prompt=prompt,
        hps_version=hps_version,
        hps_root=hps_root,
        hps_checkpoint_path=hps_checkpoint_path,
        auto_download=hps_auto_download,
    )
    return _run_test_time_oft_core(
        prompt,
        reward_model=reward_model,
        metric_keys=['hpsv2'],
        model_id=model_id,
        output_dir=output_dir,
        height=height,
        width=width,
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        weight_decay=weight_decay,
        patch_size=patch_size,
        device=device,
        seed=seed,
        variant=variant,
        save_every=save_every,
        print_eigs_every=print_eigs_every,
        save_outputs=save_outputs,
    )


def run_test_time_oft_pickscore_for_prompts(
    prompts: List[str],
    **kwargs,
) -> Dict[str, Tuple[List[torch.Tensor], Dict[str, object]]]:
    results: Dict[str, Tuple[List[torch.Tensor], Dict[str, object]]] = {}
    for i, p in enumerate(prompts):
        cur_output_dir = kwargs.get('output_dir')
        if cur_output_dir is not None and kwargs.get('save_outputs', True):
            cur_output_dir = os.path.join(cur_output_dir, f'prompt_{i}')
        local_kwargs = dict(kwargs)
        local_kwargs['output_dir'] = cur_output_dir
        images, rewards = run_test_time_oft_pickscore(p, **local_kwargs)
        results[p] = (images, rewards)
    return results


def read_prompts(path: str) -> List[str]:
    prompts: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def run_pickscore_batch_from_prompt_file(
    prompt_file: str,
    output_dir: str,
    **kwargs,
) -> Dict[str, Tuple[List[torch.Tensor], Dict[str, object]]]:
    prompts = read_prompts(prompt_file)
    if not prompts:
        raise RuntimeError(f'Prompt file is empty: {prompt_file}')

    return run_test_time_oft_pickscore_for_prompts(
        prompts,
        output_dir=output_dir,
        **kwargs,
    )


def write_pickscore_batch_metrics(
    prompts: List[str],
    results: Dict[str, Tuple[List[torch.Tensor], Dict[str, object]]],
    output_dir: str,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, prompt in enumerate(prompts):
        _, reward_dict = results[prompt]
        reward_history = [float(x) for x in reward_dict.get('reward_history', [])]
        pickscore_history = [float(x) for x in reward_dict.get('pickscore_history', [])]
        records.append(
            {
                'index': idx,
                'prompt': prompt,
                'reward': float(reward_dict.get('reward', float('nan'))),
                'pickscore': float(reward_dict.get('pickscore', float('nan'))),
                'best_reward': float(reward_dict.get('best_reward', reward_dict.get('reward', float('nan')))),
                'best_pickscore': float(reward_dict.get('best_pickscore', reward_dict.get('pickscore', float('nan')))),
                'last_reward': float(reward_dict.get('last_reward', float('nan'))),
                'last_pickscore': float(reward_dict.get('last_pickscore', float('nan'))),
                'reward_history': reward_history,
                'pickscore_history': pickscore_history,
                'num_steps': len(reward_history),
            }
        )

    metrics_path = out_dir / 'metrics_per_prompt.json'
    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    valid = [r for r in records if not torch.isnan(torch.tensor(float(r['best_reward'])))]
    if valid:
        n = len(valid)
        max_steps = max(int(r.get('num_steps', 0)) for r in valid)
        mean_reward_per_step = []
        mean_pickscore_per_step = []
        for step_idx in range(max_steps):
            step_rewards = []
            step_picks = []
            for r in valid:
                rh = r.get('reward_history', [])
                ph = r.get('pickscore_history', [])
                if step_idx < len(rh):
                    v = float(rh[step_idx])
                    if math.isfinite(v):
                        step_rewards.append(v)
                if step_idx < len(ph):
                    v = float(ph[step_idx])
                    if math.isfinite(v):
                        step_picks.append(v)
            mean_reward_per_step.append(sum(step_rewards) / len(step_rewards) if step_rewards else float('nan'))
            mean_pickscore_per_step.append(sum(step_picks) / len(step_picks) if step_picks else float('nan'))

        summary = {
            'num_samples': n,
            'mean_best_reward': sum(float(r['best_reward']) for r in valid) / n,
            'mean_best_pickscore': sum(float(r['best_pickscore']) for r in valid) / n,
            'mean_last_reward': sum(float(r['last_reward']) for r in valid) / n,
            'mean_last_pickscore': sum(float(r['last_pickscore']) for r in valid) / n,
            'mean_reward_per_step': mean_reward_per_step,
            'mean_pickscore_per_step': mean_pickscore_per_step,
        }
        summary_path = out_dir / 'metrics_summary.json'
        with summary_path.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
