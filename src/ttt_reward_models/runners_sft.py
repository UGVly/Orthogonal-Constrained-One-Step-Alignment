import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from torch.utils.data import DataLoader
from tqdm import tqdm

from .adapters import PatchOrthogonalNoise
from .data import AssignedNoiseDataset, collate_assigned_noise, copy_or_link_image, iter_pairs, load_target_image
from .diagnostics import (
    covariance_eigenvalues_from_noise,
    print_eig_report,
    print_q_spectrum,
    save_grid,
    save_noise_distribution_report,
    summarize_noise,
)
from .pipeline import OneStepSDXLTurbo, PromptCond, get_cached_sdxl_pipeline
from .utils import ensure_dir, freeze_module, get_device, set_seed



def _build_prompt_cond_batch(pipe: StableDiffusionXLPipeline, prompts: Sequence[str], height: int, width: int, device: str) -> PromptCond:
    pe, _, pooled, _ = pipe.encode_prompt(
        prompt=list(prompts),
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    text_encoder_projection_dim = None
    if getattr(pipe, 'text_encoder_2', None) is not None:
        text_encoder_projection_dim = getattr(pipe.text_encoder_2.config, 'projection_dim', None)
    add_time_ids = pipe._get_add_time_ids(
        (height, width),
        (0, 0),
        (height, width),
        dtype=pe.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    ).to(device)
    add_time_ids = add_time_ids.repeat(len(prompts), 1)
    return PromptCond(prompt_embeds=pe, pooled_prompt_embeds=pooled, add_time_ids=add_time_ids)



def _load_trainable_sdxl_pipeline(model_id: str, device: str, variant: str = 'fp16') -> StableDiffusionXLPipeline:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
        variant=variant if device.startswith('cuda') else None,
    )
    pipe = pipe.to(device)
    pipe.scheduler = pipe.scheduler.__class__.from_config(
        pipe.scheduler.config,
        timestep_spacing='trailing',
    )
    freeze_module(pipe.vae)
    if getattr(pipe, 'text_encoder', None) is not None:
        freeze_module(pipe.text_encoder)
    if getattr(pipe, 'text_encoder_2', None) is not None:
        freeze_module(pipe.text_encoder_2)
    pipe.vae.to(dtype=torch.float32)
    pipe.unet.train()
    for p in pipe.unet.parameters():
        p.requires_grad_(True)
    return pipe



def _plot_history(history: Dict[str, List[float]], output_path: str, ylabel: str = 'value') -> None:
    keys = [k for k, v in history.items() if v]
    if not keys:
        return
    xs = list(range(1, len(history[keys[0]]) + 1))
    fig, axs = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    if len(keys) == 1:
        axs = [axs]
    for ax, key in zip(axs, keys):
        vals = history[key]
        ax.plot(xs, vals)
        ax.set_title(key)
        ax.set_xlabel('step')
        ax.set_ylabel(ylabel)
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)



def run_assign_matched_noise_sft(
    prompt: str,
    *,
    target_image_path: str,
    model_id: str = 'models/sdxl-turbo',
    output_dir: Optional[str] = 'outputs/run_assign_noise_sft',
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

    target_image = load_target_image(target_image_path, height, width, dev)
    with torch.no_grad():
        target_latent = model.encode_image_to_latent(target_image)
    if batch_size > 1:
        target_image = target_image.repeat(batch_size, 1, 1, 1)
        target_latent = target_latent.repeat(batch_size, 1, 1, 1)

    if output_dir and save_outputs:
        save_grid(target_image, os.path.join(output_dir, 'target.png'))
        with torch.no_grad():
            init_eps_theta = adapter(base_noise)
            save_noise_distribution_report(
                base_noise=base_noise,
                transformed_noise=init_eps_theta,
                output_dir=output_dir,
                patch_size=patch_size,
                prefix='orthogonal_gaussian_init',
                title='Before assigned-noise optimization: base noise vs orthogonally transformed noise',
                label_a='base Gaussian noise',
                label_b='Q @ base noise (init)',
            )

    with torch.no_grad():
        init_images, _ = model(base_noise.to(dev), cond)
        if output_dir and save_outputs:
            save_grid(init_images, os.path.join(output_dir, 'init.png'))

    best_loss = float('inf')
    best_images = None
    best_input_noise = None
    best_stats: Optional[Dict[str, float]] = None
    loss_history: Dict[str, List[float]] = {
        'total_loss': [],
        'latent_loss': [],
        'pixel_l1': [],
        'drift_reg': [],
    }

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

        images, denoised = model(eps_theta, cond)
        latent_loss = F.mse_loss(denoised, target_latent)
        pixel_l1 = F.l1_loss(images, target_image) if pixel_l1_weight > 0 else torch.zeros_like(latent_loss)
        loss = latent_loss_weight * latent_loss + pixel_l1_weight * pixel_l1 + drift_reg_weight * drift_reg

        if not torch.isfinite(loss):
            raise RuntimeError(
                f'Non-finite loss at step {step}: loss={loss.item()}, latent={latent_loss.item()}, '
                f'pixel={pixel_l1.item()}, drift={drift_reg.item()}'
            )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss_val = float(loss.item())
            latent_loss_val = float(latent_loss.item())
            pixel_l1_val = float(pixel_l1.item())
            drift_reg_val = float(drift_reg.item())
            loss_history['total_loss'].append(total_loss_val)
            loss_history['latent_loss'].append(latent_loss_val)
            loss_history['pixel_l1'].append(pixel_l1_val)
            loss_history['drift_reg'].append(drift_reg_val)
            if total_loss_val < best_loss:
                best_loss = total_loss_val
                best_images = images.detach().cpu()
                best_input_noise = eps_theta.detach().cpu()
                best_stats = {
                    'total_loss': total_loss_val,
                    'latent_loss': latent_loss_val,
                    'pixel_l1': pixel_l1_val,
                    'drift_reg': drift_reg_val,
                    'best_input_noise_mean': float(best_input_noise.float().mean().item()),
                    'best_input_noise_var': float(best_input_noise.float().var(unbiased=False).item()),
                }
                if output_dir and save_outputs:
                    save_grid(best_images, os.path.join(output_dir, 'best.png'))
                    torch.save(best_input_noise, os.path.join(output_dir, 'best_input_noise.pt'))
                    torch.save(best_input_noise, os.path.join(output_dir, 'best_noise.pt'))
            if save_outputs and output_dir and (step % save_every == 0 or step == steps):
                save_grid(images, os.path.join(output_dir, f'step_{step:03d}.png'))
                torch.save(adapter.state_dict(), os.path.join(output_dir, f'adapter_step_{step:03d}.pt'))

    if output_dir and save_outputs:
        _plot_history(loss_history, os.path.join(output_dir, 'loss_history.png'), ylabel='loss')
        torch.save(adapter.state_dict(), os.path.join(output_dir, 'adapter_final.pt'))
        if best_images is not None:
            save_grid(best_images, os.path.join(output_dir, 'final_best.png'))
        final_noise = best_input_noise if best_input_noise is not None else adapter(base_noise).detach().cpu()
        save_noise_distribution_report(
            base_noise=base_noise.detach().cpu(),
            transformed_noise=final_noise,
            output_dir=output_dir,
            patch_size=patch_size,
            prefix='orthogonal_gaussian_final',
            title='After assigned-noise optimization: base noise vs orthogonally transformed noise',
            label_a='base Gaussian noise',
            label_b='Q @ base noise (final/best)',
        )
        with open(os.path.join(output_dir, 'best_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(best_stats or {}, f, indent=2, ensure_ascii=False)

    image_list: List[torch.Tensor] = []
    if best_images is not None:
        image_list = [best_images[i] for i in range(best_images.shape[0])]

    stats_out: Dict[str, object] = {
        'loss': float(best_stats['total_loss']) if best_stats is not None else float(best_loss),
        'best_loss': float(best_stats['total_loss']) if best_stats is not None else float(best_loss),
        'best_latent_loss': float(best_stats['latent_loss']) if best_stats is not None else float('nan'),
        'best_pixel_l1': float(best_stats['pixel_l1']) if best_stats is not None else float('nan'),
        'best_drift_reg': float(best_stats['drift_reg']) if best_stats is not None else float('nan'),
        'best_input_noise_mean': float(best_stats['best_input_noise_mean']) if best_stats is not None else float('nan'),
        'best_input_noise_var': float(best_stats['best_input_noise_var']) if best_stats is not None else float('nan'),
        'loss_history': [float(x) for x in loss_history['total_loss']],
        'latent_loss_history': [float(x) for x in loss_history['latent_loss']],
        'pixel_l1_history': [float(x) for x in loss_history['pixel_l1']],
        'drift_reg_history': [float(x) for x in loss_history['drift_reg']],
    }
    return image_list, stats_out



def build_assigned_noise_dataset(
    *,
    data_root: str,
    output_root: str,
    model_id: str = 'models/sdxl-turbo',
    image_field: str = 'image',
    prompt_field: str = 'prompt',
    start_index: int = 0,
    max_samples: Optional[int] = None,
    link_image: bool = True,
    skip_existing: bool = True,
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
    variant: str = 'fp16',
    save_every: int = 1,
    print_eigs_every: int = 1,
) -> Dict[str, object]:
    data_root_p = Path(data_root).resolve()
    pairs_jsonl = data_root_p / 'pairs.jsonl'
    if not pairs_jsonl.exists():
        raise FileNotFoundError(f'pairs.jsonl not found: {pairs_jsonl}')

    output_root_p = Path(output_root).resolve()
    output_root_p.mkdir(parents=True, exist_ok=True)
    samples_root = output_root_p / 'samples'
    samples_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root_p / 'manifest.jsonl'
    failures_path = output_root_p / 'failures.jsonl'

    success_count = 0
    fail_count = 0
    manifest_records: List[Dict[str, object]] = []

    for item in tqdm(
        iter_pairs(pairs_jsonl, image_field=image_field, prompt_field=prompt_field, start_index=start_index, max_samples=max_samples),
        desc='Build assigned-noise dataset',
    ):
        line_idx = int(item['_line_idx'])
        prompt = str(item[prompt_field])
        image_rel = str(item[image_field])
        image_abs = (data_root_p / image_rel).resolve()
        if not image_abs.exists():
            raise FileNotFoundError(f'Image not found for line {line_idx}: {image_abs}')
        sample_id = f'{line_idx:06d}_{Path(image_rel).stem}'
        sample_dir = samples_root / sample_id
        run_dir = sample_dir / 'run_outputs'
        meta_path = sample_dir / 'meta.json'
        if skip_existing and meta_path.exists():
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        try:
            _, stats = run_assign_matched_noise_sft(
                prompt,
                target_image_path=str(image_abs),
                model_id=model_id,
                output_dir=str(run_dir),
                height=height,
                width=width,
                batch_size=batch_size,
                steps=steps,
                lr=lr,
                weight_decay=weight_decay,
                patch_size=patch_size,
                latent_loss_weight=latent_loss_weight,
                pixel_l1_weight=pixel_l1_weight,
                drift_reg_weight=drift_reg_weight,
                device=device,
                seed=seed + line_idx,
                variant=variant,
                save_every=save_every,
                print_eigs_every=print_eigs_every,
                save_outputs=True,
            )
            best_noise_src = run_dir / 'best_input_noise.pt'
            if not best_noise_src.exists():
                best_noise_src = run_dir / 'best_noise.pt'
            if not best_noise_src.exists():
                raise FileNotFoundError(f'best input noise file missing in {run_dir}')
            target_dst = sample_dir / Path(image_rel).name
            copy_or_link_image(image_abs, target_dst, link_image=link_image)
            best_noise_dst = sample_dir / 'best_input_noise.pt'
            torch.save(torch.load(best_noise_src, map_location='cpu'), best_noise_dst)
            record: Dict[str, object] = {
                'sample_id': sample_id,
                'prompt': prompt,
                'target_image_path': str(target_dst),
                'best_input_noise_path': str(best_noise_dst),
                'source_image_path': str(image_abs),
                'source_rel_image_path': image_rel,
                'line_idx': line_idx,
                'stats': stats,
            }
            with meta_path.open('w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            manifest_records.append(record)
            success_count += 1
        except Exception as exc:  # noqa: BLE001
            fail_count += 1
            failure = {
                'line_idx': line_idx,
                'prompt': prompt,
                'image': image_rel,
                'error': repr(exc),
            }
            with failures_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(failure, ensure_ascii=False) + '\n')

    manifest_records.sort(key=lambda x: int(x['line_idx']))
    with manifest_path.open('w', encoding='utf-8') as f:
        for rec in manifest_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    summary = {
        'manifest_path': str(manifest_path),
        'failures_path': str(failures_path),
        'success_count': success_count,
        'fail_count': fail_count,
    }
    with (output_root_p / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary



def train_latent_matched_sft(
    *,
    manifest_path: str,
    model_id: str = 'models/sdxl-turbo',
    output_dir: str = 'outputs/latent_matched_sft',
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
    epochs: int = 1,
    lr: float = 1e-5,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    latent_loss_weight: float = 1.0,
    pixel_l1_weight: float = 0.0,
    preserve_latent_weight: float = 0.0,
    preserve_pixel_weight: float = 0.0,
    preserve_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    seed: int = 42,
    variant: str = 'fp16',
    num_workers: int = 0,
    save_every_epochs: int = 1,
) -> Dict[str, object]:
    ensure_dir(output_dir)
    set_seed(seed)
    dev = get_device(device)
    torch.set_grad_enabled(True)

    dataset = AssignedNoiseDataset(manifest_path, height=height, width=width)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=dev.startswith('cuda'),
        collate_fn=collate_assigned_noise,
    )

    train_pipe = _load_trainable_sdxl_pipeline(model_id=model_id, device=dev, variant=variant)
    student = OneStepSDXLTurbo(train_pipe, height=height, width=width)
    teacher = None
    teacher_model = None
    if preserve_latent_weight > 0 or preserve_pixel_weight > 0:
        teacher = get_cached_sdxl_pipeline(model_id=model_id, device=dev, variant=variant)
        teacher_model = OneStepSDXLTurbo(teacher, height=height, width=width)

    optimizer = torch.optim.AdamW(train_pipe.unet.parameters(), lr=lr, weight_decay=weight_decay)
    history: Dict[str, List[float]] = {
        'total_loss': [],
        'recon_latent': [],
        'recon_pixel_l1': [],
        'preserve_latent': [],
        'preserve_pixel': [],
    }
    best_loss = float('inf')
    best_epoch = -1
    gen_device = dev if dev.startswith('cuda') else 'cpu'
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)

    for epoch in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f'Latent-matched SFT epoch {epoch}/{epochs}')
        for batch in pbar:
            prompts = list(batch['prompts'])
            target_images = batch['images'].to(dev)
            matched_noises = batch['matched_noises'].to(dev)
            optimizer.zero_grad(set_to_none=True)

            cond = _build_prompt_cond_batch(train_pipe, prompts, height, width, dev)
            images_pred, denoised_pred = student(matched_noises, cond)
            with torch.no_grad():
                target_latent = student.encode_image_to_latent(target_images)

            recon_latent = F.mse_loss(denoised_pred, target_latent)
            recon_pixel = F.l1_loss(images_pred, target_images) if pixel_l1_weight > 0 else torch.zeros_like(recon_latent)
            loss = latent_loss_weight * recon_latent + pixel_l1_weight * recon_pixel

            preserve_latent = torch.zeros_like(recon_latent)
            preserve_pixel = torch.zeros_like(recon_latent)
            if teacher_model is not None:
                pb = preserve_batch_size or len(prompts)
                rand_noises = student.sample_base_noise(pb, generator=generator)
                preserve_prompts = prompts[:pb]
                cond_student = _build_prompt_cond_batch(train_pipe, preserve_prompts, height, width, dev)
                cond_teacher = _build_prompt_cond_batch(teacher, preserve_prompts, height, width, dev)
                student_imgs_r, student_lat_r = student(rand_noises, cond_student)
                with torch.no_grad():
                    teacher_imgs_r, teacher_lat_r = teacher_model(rand_noises, cond_teacher)
                if preserve_latent_weight > 0:
                    preserve_latent = F.mse_loss(student_lat_r, teacher_lat_r)
                    loss = loss + preserve_latent_weight * preserve_latent
                if preserve_pixel_weight > 0:
                    preserve_pixel = F.l1_loss(student_imgs_r, teacher_imgs_r)
                    loss = loss + preserve_pixel_weight * preserve_pixel

            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite SFT loss: {loss.item()}')
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(train_pipe.unet.parameters(), max_grad_norm)
            optimizer.step()

            total_val = float(loss.item())
            history['total_loss'].append(total_val)
            history['recon_latent'].append(float(recon_latent.item()))
            history['recon_pixel_l1'].append(float(recon_pixel.item()))
            history['preserve_latent'].append(float(preserve_latent.item()))
            history['preserve_pixel'].append(float(preserve_pixel.item()))
            pbar.set_postfix(total=f'{total_val:.4f}', recon=f'{float(recon_latent.item()):.4f}')

        epoch_mean_loss = sum(history['total_loss'][-len(loader):]) / max(len(loader), 1)
        if epoch_mean_loss < best_loss:
            best_loss = epoch_mean_loss
            best_epoch = epoch
            best_dir = os.path.join(output_dir, 'checkpoint-best')
            ensure_dir(best_dir)
            train_pipe.save_pretrained(best_dir)
        if save_every_epochs > 0 and epoch % save_every_epochs == 0:
            ckpt_dir = os.path.join(output_dir, f'checkpoint-epoch-{epoch:03d}')
            ensure_dir(ckpt_dir)
            train_pipe.save_pretrained(ckpt_dir)

    final_dir = os.path.join(output_dir, 'checkpoint-final')
    ensure_dir(final_dir)
    train_pipe.save_pretrained(final_dir)
    _plot_history(history, os.path.join(output_dir, 'train_history.png'), ylabel='loss')
    summary: Dict[str, object] = {
        'manifest_path': manifest_path,
        'num_samples': len(dataset),
        'epochs': epochs,
        'best_epoch': best_epoch,
        'best_epoch_mean_loss': best_loss,
        'final_checkpoint': final_dir,
        'best_checkpoint': os.path.join(output_dir, 'checkpoint-best'),
        'history': history,
    }
    with open(os.path.join(output_dir, 'train_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary



def run_em_latent_matched_sft(
    *,
    data_root: str,
    work_dir: str,
    initial_model_id: str = 'models/sdxl-turbo',
    num_cycles: int = 2,
    image_field: str = 'image',
    prompt_field: str = 'prompt',
    height: int = 512,
    width: int = 512,
    assign_steps: int = 30,
    assign_lr: float = 5e-4,
    train_epochs: int = 1,
    train_batch_size: int = 1,
    train_lr: float = 1e-5,
    latent_loss_weight: float = 1.0,
    pixel_l1_weight: float = 0.0,
    preserve_latent_weight: float = 0.0,
    preserve_pixel_weight: float = 0.0,
    patch_size: int = 2,
    device: Optional[str] = None,
    seed: int = 42,
    variant: str = 'fp16',
) -> Dict[str, object]:
    ensure_dir(work_dir)
    current_model_id = initial_model_id
    cycle_records: List[Dict[str, object]] = []
    for cycle in range(1, num_cycles + 1):
        cycle_dir = os.path.join(work_dir, f'cycle_{cycle:02d}')
        assign_dir = os.path.join(cycle_dir, 'assigned_dataset')
        train_dir = os.path.join(cycle_dir, 'train')
        assign_summary = build_assigned_noise_dataset(
            data_root=data_root,
            output_root=assign_dir,
            model_id=current_model_id,
            image_field=image_field,
            prompt_field=prompt_field,
            height=height,
            width=width,
            steps=assign_steps,
            lr=assign_lr,
            patch_size=patch_size,
            latent_loss_weight=latent_loss_weight,
            pixel_l1_weight=pixel_l1_weight,
            device=device,
            seed=seed + 1000 * cycle,
            variant=variant,
        )
        train_summary = train_latent_matched_sft(
            manifest_path=assign_summary['manifest_path'],
            model_id=current_model_id,
            output_dir=train_dir,
            height=height,
            width=width,
            batch_size=train_batch_size,
            epochs=train_epochs,
            lr=train_lr,
            latent_loss_weight=latent_loss_weight,
            pixel_l1_weight=pixel_l1_weight,
            preserve_latent_weight=preserve_latent_weight,
            preserve_pixel_weight=preserve_pixel_weight,
            device=device,
            seed=seed + 2000 * cycle,
            variant=variant,
        )
        current_model_id = str(train_summary['final_checkpoint'])
        cycle_records.append({
            'cycle': cycle,
            'assigned_dataset': assign_summary,
            'train': train_summary,
            'next_model_id': current_model_id,
        })
    summary = {
        'initial_model_id': initial_model_id,
        'final_model_id': current_model_id,
        'num_cycles': num_cycles,
        'cycles': cycle_records,
    }
    with open(os.path.join(work_dir, 'em_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
