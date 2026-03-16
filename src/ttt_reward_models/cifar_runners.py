from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .cifar_data import AssignedCIFAR10Dataset, CIFAR10TargetDataset, assigned_cifar_collate, save_tensor_image
from .cifar_models import CIFARConditionalDiscriminator, CIFARConditionalGenerator, sample_cifar_grid
from .diagnostics import save_grid
from .orthogonal_latent import optimize_orthogonal_noise_for_target
from .utils import ensure_dir, get_device, set_seed


def _save_history_plot(history: Dict[str, List[float]], path: str, ylabel: str = 'loss') -> None:
    keys = [k for k, v in history.items() if v]
    if not keys:
        return
    fig, axs = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    if len(keys) == 1:
        axs = [axs]
    for ax, key in zip(axs, keys):
        xs = list(range(1, len(history[key]) + 1))
        ax.plot(xs, history[key])
        ax.set_title(key)
        ax.set_xlabel('step')
        ax.set_ylabel(ylabel)
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def _hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def _load_generator_checkpoint(generator: CIFARConditionalGenerator, checkpoint_path: str, device: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['generator'] if isinstance(ckpt, dict) and 'generator' in ckpt else ckpt
    generator.load_state_dict(state)


@torch.no_grad()
def _save_eval_grid(generator: CIFARConditionalGenerator, output_path: str, device: str, z: Optional[torch.Tensor] = None) -> None:
    grid = sample_cifar_grid(generator, device=device, z=z)
    save_grid(grid, output_path)


def _make_teacher(generator_ckpt: str, z_dim: int, device: str) -> CIFARConditionalGenerator:
    teacher = CIFARConditionalGenerator(z_dim=z_dim).to(device)
    _load_generator_checkpoint(teacher, generator_ckpt, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def train_cifar_one_step_gan(
    *,
    output_dir: str,
    data_root: str = './data',
    batch_size: int = 128,
    epochs: int = 20,
    z_dim: int = 128,
    generator_lr: float = 2e-4,
    discriminator_lr: float = 2e-4,
    num_workers: int = 2,
    device: Optional[str] = None,
    seed: int = 42,
    train_limit: Optional[int] = None,
    eval_per_class: int = 8,
) -> None:
    ensure_dir(output_dir)
    set_seed(seed)
    dev = get_device(device)

    dataset = CIFAR10TargetDataset(root=data_root, train=True, limit=train_limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    generator = CIFARConditionalGenerator(z_dim=z_dim).to(dev)
    discriminator = CIFARConditionalDiscriminator().to(dev)
    opt_g = torch.optim.Adam(generator.parameters(), lr=generator_lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))

    fixed_labels = torch.arange(10, device=dev).repeat_interleave(eval_per_class)
    fixed_z = torch.randn(fixed_labels.shape[0], z_dim, device=dev)
    history: Dict[str, List[float]] = {'d_loss': [], 'g_loss': []}

    for epoch in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f'CIFAR GAN epoch {epoch}/{epochs}')
        for batch in pbar:
            real = batch['image'].to(dev)
            labels = batch['label'].to(dev) if isinstance(batch['label'], torch.Tensor) else torch.tensor(batch['label'], dtype=torch.long, device=dev)
            bs = real.shape[0]
            z = torch.randn(bs, z_dim, device=dev)

            fake = generator(z, labels)
            opt_d.zero_grad(set_to_none=True)
            d_loss = _hinge_d_loss(discriminator(real, labels), discriminator(fake.detach(), labels))
            d_loss.backward()
            opt_d.step()

            z2 = torch.randn(bs, z_dim, device=dev)
            fake2 = generator(z2, labels)
            opt_g.zero_grad(set_to_none=True)
            g_loss = _hinge_g_loss(discriminator(fake2, labels))
            g_loss.backward()
            opt_g.step()

            history['d_loss'].append(float(d_loss.item()))
            history['g_loss'].append(float(g_loss.item()))
            pbar.set_postfix(d_loss=f'{d_loss.item():.4f}', g_loss=f'{g_loss.item():.4f}')

        _save_eval_grid(generator, os.path.join(output_dir, f'epoch_{epoch:03d}_samples.png'), dev, z=fixed_z)
        ckpt = {
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'epoch': epoch,
            'z_dim': z_dim,
        }
        torch.save(ckpt, os.path.join(output_dir, f'cifar_gan_epoch_{epoch:03d}.pt'))

    torch.save({'generator': generator.state_dict(), 'discriminator': discriminator.state_dict(), 'z_dim': z_dim}, os.path.join(output_dir, 'cifar_gan_final.pt'))
    _save_history_plot(history, os.path.join(output_dir, 'cifar_gan_loss_history.png'))
    _save_eval_grid(generator, os.path.join(output_dir, 'cifar_gan_final_samples.png'), dev, z=fixed_z)


def build_cifar_assigned_noise_dataset(
    *,
    generator_ckpt: str,
    output_dir: str,
    data_root: str = './data',
    train_split: bool = True,
    max_items: int = 512,
    assign_steps: int = 300,
    assign_lr: float = 5e-2,
    z_dim: int = 128,
    mse_weight: float = 1.0,
    l1_weight: float = 0.0,
    prior_weight: float = 1e-3,
    device: Optional[str] = None,
    seed: int = 42,
) -> None:
    ensure_dir(output_dir)
    images_dir = os.path.join(output_dir, 'target_images')
    recons_dir = os.path.join(output_dir, 'assigned_recons')
    noises_dir = os.path.join(output_dir, 'assigned_noises')
    base_noises_dir = os.path.join(output_dir, 'assigned_base_noises')
    orth_dir = os.path.join(output_dir, 'assigned_orthogonal_maps')
    curves_dir = os.path.join(output_dir, 'assign_curves')
    ensure_dir(images_dir)
    ensure_dir(recons_dir)
    ensure_dir(noises_dir)
    ensure_dir(base_noises_dir)
    ensure_dir(orth_dir)
    ensure_dir(curves_dir)

    set_seed(seed)
    dev = get_device(device)
    dataset = CIFAR10TargetDataset(root=data_root, train=train_split, limit=max_items)

    generator = CIFARConditionalGenerator(z_dim=z_dim).to(dev)
    _load_generator_checkpoint(generator, generator_ckpt, dev)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)

    manifest_path = os.path.join(output_dir, 'manifest.jsonl')
    history: Dict[str, List[float]] = {'best_loss': []}

    with open(manifest_path, 'w', encoding='utf-8') as manifest_f:
        for item_idx in tqdm(range(len(dataset)), desc='Assign matched CIFAR noise'):
            item = dataset[item_idx]
            target = item['image'].unsqueeze(0).to(dev)
            label = torch.tensor([item['label']], dtype=torch.long, device=dev)
            result = optimize_orthogonal_noise_for_target(
                generator=generator,
                target=target,
                label=label,
                z_dim=z_dim,
                assign_steps=assign_steps,
                assign_lr=assign_lr,
                mse_weight=mse_weight,
                l1_weight=l1_weight,
                prior_weight=prior_weight,
            )
            best_loss = result.best_loss
            best_z = result.best_noise
            best_base_noise = result.best_base_noise
            best_q = result.best_q
            best_recon = result.best_recon
            curve = result.curve

            history['best_loss'].append(best_loss)
            target_path = os.path.join(images_dir, f'{item_idx:06d}_target.png')
            recon_path = os.path.join(recons_dir, f'{item_idx:06d}_recon.png')
            noise_path = os.path.join(noises_dir, f'{item_idx:06d}_noise.pt')
            base_noise_path = os.path.join(base_noises_dir, f'{item_idx:06d}_base_noise.pt')
            orth_path = os.path.join(orth_dir, f'{item_idx:06d}_orth.pt')
            curve_path = os.path.join(curves_dir, f'{item_idx:06d}_curve.png')
            save_tensor_image(item['image'], target_path)
            save_tensor_image(best_recon, recon_path)
            torch.save(best_z[0], noise_path)
            torch.save(best_base_noise[0], base_noise_path)
            torch.save(best_q, orth_path)

            plt.figure(figsize=(5, 4))
            plt.plot(curve)
            plt.xlabel('step')
            plt.ylabel('assign loss')
            plt.title(f'CIFAR assign #{item_idx}')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(curve_path, dpi=160)
            plt.close()

            record = {
                'index': item_idx,
                'source_index': int(item['source_index']),
                'label': int(item['label']),
                'prompt': str(item['prompt']),
                'target_image_path': target_path,
                'assigned_recon_path': recon_path,
                'noise_path': noise_path,
                'base_noise_path': base_noise_path,
                'orthogonal_map_path': orth_path,
                'assign_curve_path': curve_path,
                'best_loss': best_loss,
                'assignment_mode': 'orthogonal_latent_transform',
                'orthogonality_fro': float(result.metrics['orthogonality_fro']),
                'sv_min': float(result.metrics['sv_min']),
                'sv_max': float(result.metrics['sv_max']),
                'norm_diff_abs': float(result.metrics['norm_diff_abs']),
            }
            manifest_f.write(json.dumps(record, ensure_ascii=False) + '\n')

    _save_history_plot(history, os.path.join(output_dir, 'assign_best_loss_history.png'))
    with open(os.path.join(output_dir, 'assign_summary.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'num_items': len(dataset),
            'mean_best_loss': float(sum(history['best_loss']) / max(len(history['best_loss']), 1)),
            'max_best_loss': float(max(history['best_loss']) if history['best_loss'] else 0.0),
            'min_best_loss': float(min(history['best_loss']) if history['best_loss'] else 0.0),
            'manifest_path': manifest_path,
            'assignment_mode': 'orthogonal_latent_transform',
            'note': 'Each matched latent is constructed as z = Q epsilon with Q orthogonal and epsilon sampled from a standard Gaussian.',
        }, f, indent=2, ensure_ascii=False)


def _save_recon_grid(generator: CIFARConditionalGenerator, dataset: AssignedCIFAR10Dataset, output_path: str, device: str, num_items: int = 16) -> None:
    generator.eval()
    limit = min(len(dataset), num_items)
    target_list = []
    pred_list = []
    for idx in range(limit):
        item = dataset[idx]
        target = item['image'].unsqueeze(0).to(device)
        label = torch.tensor([item['label']], dtype=torch.long, device=device)
        z = item['noise'].unsqueeze(0).to(device)
        with torch.no_grad():
            recon = generator(z, label)
        target_list.append((target.detach().cpu() + 1.0) * 0.5)
        pred_list.append((recon.detach().cpu() + 1.0) * 0.5)
    stacked = []
    for t, p in zip(target_list, pred_list):
        stacked.extend([t[0], p[0]])
    save_grid(torch.stack(stacked, dim=0), output_path)
    generator.train()


def train_cifar_latent_matched_sft(
    *,
    generator_ckpt: str,
    manifest_path: str,
    output_dir: str,
    batch_size: int = 64,
    epochs: int = 5,
    lr: float = 1e-4,
    z_dim: int = 128,
    mse_weight: float = 1.0,
    l1_weight: float = 0.0,
    preserve_weight: float = 0.25,
    num_workers: int = 0,
    device: Optional[str] = None,
    seed: int = 42,
) -> None:
    ensure_dir(output_dir)
    set_seed(seed)
    dev = get_device(device)

    dataset = AssignedCIFAR10Dataset(manifest_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=assigned_cifar_collate)

    student = CIFARConditionalGenerator(z_dim=z_dim).to(dev)
    _load_generator_checkpoint(student, generator_ckpt, dev)
    teacher = _make_teacher(generator_ckpt, z_dim=z_dim, device=dev)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, betas=(0.9, 0.999))

    fixed_labels = torch.arange(10, device=dev).repeat_interleave(8)
    fixed_z = torch.randn(fixed_labels.shape[0], z_dim, device=dev)
    history: Dict[str, List[float]] = {'total_loss': [], 'target_loss': [], 'preserve_loss': []}

    for epoch in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f'CIFAR latent-matched SFT epoch {epoch}/{epochs}')
        for batch in pbar:
            target = batch['image'].to(dev)
            labels = batch['label'].to(dev)
            z = batch['noise'].to(dev)
            if z.ndim == 1:
                z = z.unsqueeze(0)

            optimizer.zero_grad(set_to_none=True)
            pred = student(z, labels)
            target_mse = F.mse_loss(pred, target)
            target_l1 = F.l1_loss(pred, target) if l1_weight > 0 else torch.zeros_like(target_mse)
            target_loss = mse_weight * target_mse + l1_weight * target_l1

            rand_z = torch.randn(labels.shape[0], z_dim, device=dev)
            with torch.no_grad():
                teacher_out = teacher(rand_z, labels)
            student_out = student(rand_z, labels)
            preserve_loss = F.mse_loss(student_out, teacher_out)
            loss = target_loss + preserve_weight * preserve_loss
            loss.backward()
            optimizer.step()

            history['total_loss'].append(float(loss.item()))
            history['target_loss'].append(float(target_loss.item()))
            history['preserve_loss'].append(float(preserve_loss.item()))
            pbar.set_postfix(total=f'{loss.item():.4f}', target=f'{target_loss.item():.4f}', preserve=f'{preserve_loss.item():.4f}')

        _save_eval_grid(student, os.path.join(output_dir, f'epoch_{epoch:03d}_random_samples.png'), dev, z=fixed_z)
        _save_recon_grid(student, dataset, os.path.join(output_dir, f'epoch_{epoch:03d}_assigned_recons.png'), dev)
        torch.save({'generator': student.state_dict(), 'epoch': epoch, 'z_dim': z_dim}, os.path.join(output_dir, f'cifar_latent_matched_sft_epoch_{epoch:03d}.pt'))

    torch.save({'generator': student.state_dict(), 'z_dim': z_dim}, os.path.join(output_dir, 'cifar_latent_matched_sft_final.pt'))
    _save_history_plot(history, os.path.join(output_dir, 'cifar_latent_matched_sft_loss_history.png'))
    _save_eval_grid(student, os.path.join(output_dir, 'cifar_latent_matched_sft_final_random_samples.png'), dev, z=fixed_z)
    _save_recon_grid(student, dataset, os.path.join(output_dir, 'cifar_latent_matched_sft_final_assigned_recons.png'), dev)


def train_cifar_direct_random_noise_sft(
    *,
    generator_ckpt: str,
    data_root: str,
    output_dir: str,
    batch_size: int = 64,
    epochs: int = 5,
    lr: float = 1e-4,
    z_dim: int = 128,
    mse_weight: float = 1.0,
    l1_weight: float = 0.0,
    preserve_weight: float = 0.0,
    num_workers: int = 0,
    train_limit: Optional[int] = 4096,
    device: Optional[str] = None,
    seed: int = 42,
) -> None:
    ensure_dir(output_dir)
    set_seed(seed)
    dev = get_device(device)

    dataset = CIFAR10TargetDataset(root=data_root, train=True, limit=train_limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    student = CIFARConditionalGenerator(z_dim=z_dim).to(dev)
    _load_generator_checkpoint(student, generator_ckpt, dev)
    teacher = _make_teacher(generator_ckpt, z_dim=z_dim, device=dev)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, betas=(0.9, 0.999))

    fixed_labels = torch.arange(10, device=dev).repeat_interleave(8)
    fixed_z = torch.randn(fixed_labels.shape[0], z_dim, device=dev)
    history: Dict[str, List[float]] = {'total_loss': [], 'target_loss': [], 'preserve_loss': []}

    for epoch in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f'CIFAR direct random-noise SFT epoch {epoch}/{epochs}')
        for batch in pbar:
            target = batch['image'].to(dev)
            labels = batch['label'].to(dev) if isinstance(batch['label'], torch.Tensor) else torch.tensor(batch['label'], dtype=torch.long, device=dev)
            bs = target.shape[0]
            z = torch.randn(bs, z_dim, device=dev)

            optimizer.zero_grad(set_to_none=True)
            pred = student(z, labels)
            target_mse = F.mse_loss(pred, target)
            target_l1 = F.l1_loss(pred, target) if l1_weight > 0 else torch.zeros_like(target_mse)
            target_loss = mse_weight * target_mse + l1_weight * target_l1

            preserve_loss = torch.zeros_like(target_loss)
            if preserve_weight > 0:
                rand_z = torch.randn(bs, z_dim, device=dev)
                with torch.no_grad():
                    teacher_out = teacher(rand_z, labels)
                student_out = student(rand_z, labels)
                preserve_loss = F.mse_loss(student_out, teacher_out)
            loss = target_loss + preserve_weight * preserve_loss
            loss.backward()
            optimizer.step()

            history['total_loss'].append(float(loss.item()))
            history['target_loss'].append(float(target_loss.item()))
            history['preserve_loss'].append(float(preserve_loss.item()))
            pbar.set_postfix(total=f'{loss.item():.4f}', target=f'{target_loss.item():.4f}', preserve=f'{preserve_loss.item():.4f}')

        _save_eval_grid(student, os.path.join(output_dir, f'epoch_{epoch:03d}_random_samples.png'), dev, z=fixed_z)
        torch.save({'generator': student.state_dict(), 'epoch': epoch, 'z_dim': z_dim}, os.path.join(output_dir, f'cifar_direct_sft_epoch_{epoch:03d}.pt'))

    torch.save({'generator': student.state_dict(), 'z_dim': z_dim}, os.path.join(output_dir, 'cifar_direct_sft_final.pt'))
    _save_history_plot(history, os.path.join(output_dir, 'cifar_direct_sft_loss_history.png'))
    _save_eval_grid(student, os.path.join(output_dir, 'cifar_direct_sft_final_random_samples.png'), dev, z=fixed_z)


def run_cifar_em_latent_matched_sft(
    *,
    generator_ckpt: str,
    data_root: str,
    output_dir: str,
    em_rounds: int = 3,
    assign_max_items: int = 512,
    assign_steps: int = 300,
    assign_lr: float = 5e-2,
    sft_epochs: int = 2,
    sft_batch_size: int = 64,
    sft_lr: float = 1e-4,
    z_dim: int = 128,
    device: Optional[str] = None,
    seed: int = 42,
) -> None:
    ensure_dir(output_dir)
    current_ckpt = generator_ckpt
    for round_idx in range(1, em_rounds + 1):
        round_dir = os.path.join(output_dir, f'round_{round_idx:02d}')
        assign_dir = os.path.join(round_dir, 'assigned_dataset')
        sft_dir = os.path.join(round_dir, 'sft')
        build_cifar_assigned_noise_dataset(
            generator_ckpt=current_ckpt,
            output_dir=assign_dir,
            data_root=data_root,
            train_split=True,
            max_items=assign_max_items,
            assign_steps=assign_steps,
            assign_lr=assign_lr,
            z_dim=z_dim,
            device=device,
            seed=seed + round_idx,
        )
        train_cifar_latent_matched_sft(
            generator_ckpt=current_ckpt,
            manifest_path=os.path.join(assign_dir, 'manifest.jsonl'),
            output_dir=sft_dir,
            batch_size=sft_batch_size,
            epochs=sft_epochs,
            lr=sft_lr,
            z_dim=z_dim,
            device=device,
            seed=seed + round_idx,
        )
        current_ckpt = os.path.join(sft_dir, 'cifar_latent_matched_sft_final.pt')

    final_summary = {
        'initial_checkpoint': generator_ckpt,
        'final_checkpoint': current_ckpt,
        'em_rounds': em_rounds,
    }
    with open(os.path.join(output_dir, 'em_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)
