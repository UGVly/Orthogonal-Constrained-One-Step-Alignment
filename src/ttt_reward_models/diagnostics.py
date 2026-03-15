import json
import math
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch


def noise_to_patch_matrix(noise: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, h, w = noise.shape
    p = patch_size
    if h % p != 0 or w % p != 0:
        raise ValueError(f"Noise shape {(h, w)} must be divisible by patch_size={p}.")
    x = noise.detach().view(b, c, h // p, p, w // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.view(-1, c * p * p)
    return x


def covariance_matrix_from_patch_matrix(x: torch.Tensor, center: bool = True) -> torch.Tensor:
    x = x.detach().to(dtype=torch.float64, device='cpu')
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    n = x.shape[0]
    cov = (x.T @ x) / max(n - 1, 1)
    cov = 0.5 * (cov + cov.T)
    return cov


def covariance_eigenvalues_from_noise(noise: torch.Tensor, patch_size: int = 2, center: bool = True) -> torch.Tensor:
    x = noise_to_patch_matrix(noise.detach(), patch_size=patch_size)
    cov = covariance_matrix_from_patch_matrix(x, center=center)
    return torch.linalg.eigvalsh(cov)


def summarize_noise(noise: torch.Tensor, name: str) -> Dict[str, float]:
    flat = noise.detach().float().reshape(-1).cpu()
    summary = {
        'mean': float(flat.mean().item()),
        'std': float(flat.std(unbiased=True).item()) if flat.numel() > 1 else 0.0,
        'var': float(flat.var(unbiased=True).item()) if flat.numel() > 1 else 0.0,
        'min': float(flat.min().item()),
        'max': float(flat.max().item()),
        'l2': float(torch.linalg.vector_norm(flat).item()),
        'numel': int(flat.numel()),
    }
    print(
        f"[{name}] mean={summary['mean']:+.6f}, std={summary['std']:.6f}, var={summary['var']:.6f}, "
        f"min={summary['min']:+.6f}, max={summary['max']:+.6f}, l2={summary['l2']:.6f}, numel={summary['numel']}"
    )
    return summary


def print_eig_report(step: int, base_eigs: torch.Tensor, cur_eigs: torch.Tensor) -> None:
    base = base_eigs.detach().cpu().to(torch.float64)
    cur = cur_eigs.detach().cpu().to(torch.float64)
    eig_diff = (cur - base).abs()
    print(
        f"[step {step:03d}] cov eigs | base min/max=({base.min().item():.6f}, {base.max().item():.6f}) | "
        f"cur min/max=({cur.min().item():.6f}, {cur.max().item():.6f}) | mean|Δ|={eig_diff.mean().item():.6e}, "
        f"max|Δ|={eig_diff.max().item():.6e}"
    )


def print_q_spectrum(step: int, q: torch.Tensor) -> None:
    q_cpu = q.detach().cpu().to(torch.float64)
    s = torch.linalg.svdvals(q_cpu)
    eye = torch.eye(q_cpu.shape[0], dtype=q_cpu.dtype)
    ortho_err = torch.linalg.matrix_norm(q_cpu.T @ q_cpu - eye).item()
    print(
        f"[step {step:03d}] Q spectrum | σ_min={s.min().item():.6f}, σ_max={s.max().item():.6f}, "
        f"meanσ={s.mean().item():.6f}, ||Q^TQ-I||_F={ortho_err:.6e}"
    )


def _summary_from_patch_matrix(x: torch.Tensor) -> Dict[str, object]:
    x_cpu = x.detach().float().cpu()
    flat = x_cpu.reshape(-1)
    mean_per_dim = x_cpu.mean(dim=0)
    var_per_dim = x_cpu.var(dim=0, unbiased=True) if x_cpu.shape[0] > 1 else torch.zeros(x_cpu.shape[1])
    norms = torch.linalg.vector_norm(x_cpu, dim=1)
    cov = covariance_matrix_from_patch_matrix(x_cpu, center=True)
    eigs = torch.linalg.eigvalsh(cov).to(torch.float32)
    eye = torch.eye(cov.shape[0], dtype=cov.dtype)
    return {
        'flat': flat,
        'mean_per_dim': mean_per_dim,
        'var_per_dim': var_per_dim,
        'norms': norms,
        'cov_eigs': eigs,
        'overall_mean': float(flat.mean().item()),
        'overall_var': float(flat.var(unbiased=True).item()) if flat.numel() > 1 else 0.0,
        'overall_std': float(flat.std(unbiased=True).item()) if flat.numel() > 1 else 0.0,
        'num_samples': int(x_cpu.shape[0]),
        'dim': int(x_cpu.shape[1]),
        'mean_abs_mean_per_dim': float(mean_per_dim.abs().mean().item()),
        'max_abs_mean_per_dim': float(mean_per_dim.abs().max().item()),
        'mean_abs_var_minus_1': float((var_per_dim - 1.0).abs().mean().item()),
        'max_abs_var_minus_1': float((var_per_dim - 1.0).abs().max().item()),
        'cov_identity_fro': float(torch.linalg.matrix_norm(cov - eye).item()),
        'cov_trace': float(torch.trace(cov).item()),
        'norm_mean': float(norms.mean().item()),
        'norm_std': float(norms.std(unbiased=True).item()) if norms.numel() > 1 else 0.0,
    }


def compare_patch_distributions(base_x: torch.Tensor, transformed_x: torch.Tensor) -> Dict[str, object]:
    base_stats = _summary_from_patch_matrix(base_x)
    transformed_stats = _summary_from_patch_matrix(transformed_x)
    compare = {
        'mean_abs_diff_mean_per_dim': float((base_stats['mean_per_dim'] - transformed_stats['mean_per_dim']).abs().mean().item()),
        'max_abs_diff_mean_per_dim': float((base_stats['mean_per_dim'] - transformed_stats['mean_per_dim']).abs().max().item()),
        'mean_abs_diff_var_per_dim': float((base_stats['var_per_dim'] - transformed_stats['var_per_dim']).abs().mean().item()),
        'max_abs_diff_var_per_dim': float((base_stats['var_per_dim'] - transformed_stats['var_per_dim']).abs().max().item()),
        'mean_abs_diff_cov_eigs': float((base_stats['cov_eigs'] - transformed_stats['cov_eigs']).abs().mean().item()),
        'max_abs_diff_cov_eigs': float((base_stats['cov_eigs'] - transformed_stats['cov_eigs']).abs().max().item()),
        'overall_mean_abs_diff': abs(base_stats['overall_mean'] - transformed_stats['overall_mean']),
        'overall_var_abs_diff': abs(base_stats['overall_var'] - transformed_stats['overall_var']),
    }
    return {'base': base_stats, 'transformed': transformed_stats, 'compare': compare}


def compare_noises(base_noise: torch.Tensor, transformed_noise: torch.Tensor, patch_size: int) -> Dict[str, object]:
    return compare_patch_distributions(noise_to_patch_matrix(base_noise, patch_size), noise_to_patch_matrix(transformed_noise, patch_size))


def _plot_dim_series(ax, y_a: torch.Tensor, y_b: torch.Tensor, title: str, label_a: str, label_b: str) -> None:
    idx = torch.arange(y_a.numel())
    ax.plot(idx.tolist(), y_a.tolist(), linewidth=1.8, label=label_a)
    ax.plot(idx.tolist(), y_b.tolist(), linewidth=1.8, label=label_b)
    ax.set_title(title)
    ax.set_xlabel('dimension index')
    ax.grid(True)
    ax.legend()


def _plot_hist(ax, a: torch.Tensor, b: torch.Tensor, title: str, label_a: str, label_b: str, bins: int = 80) -> None:
    ax.hist(a.tolist(), bins=bins, alpha=0.6, density=True, label=label_a)
    ax.hist(b.tolist(), bins=bins, alpha=0.6, density=True, label=label_b)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


def save_patch_distribution_report(base_x: torch.Tensor, transformed_x: torch.Tensor, output_dir: str, prefix: str, title: Optional[str] = None, label_a: str = 'base Gaussian', label_b: str = 'Q @ Gaussian') -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    report = compare_patch_distributions(base_x, transformed_x)
    base = report['base']
    transformed = report['transformed']
    compare = report['compare']
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    _plot_hist(axs[0, 0], base['flat'], transformed['flat'], 'All scalar values', label_a, label_b)
    _plot_dim_series(axs[0, 1], base['mean_per_dim'], transformed['mean_per_dim'], 'Per-dim means', label_a, label_b)
    _plot_dim_series(axs[0, 2], base['var_per_dim'], transformed['var_per_dim'], 'Per-dim variances', label_a, label_b)
    _plot_hist(axs[1, 0], base['norms'], transformed['norms'], 'Patch-vector norms', label_a, label_b)
    _plot_dim_series(axs[1, 1], torch.sort(base['cov_eigs']).values, torch.sort(transformed['cov_eigs']).values, 'Sorted covariance eigenvalues', label_a, label_b)
    axs[1, 2].axis('off')
    text = (
        f"{label_a}: mean={base['overall_mean']:+.4e}, var={base['overall_var']:.4f}, E|μ_d|={base['mean_abs_mean_per_dim']:.4e}, E|σ_d²-1|={base['mean_abs_var_minus_1']:.4e}\n"
        f"{label_b}: mean={transformed['overall_mean']:+.4e}, var={transformed['overall_var']:.4f}, E|μ_d|={transformed['mean_abs_mean_per_dim']:.4e}, E|σ_d²-1|={transformed['mean_abs_var_minus_1']:.4e}\n"
        f"Δ means per dim: mean={compare['mean_abs_diff_mean_per_dim']:.4e}, max={compare['max_abs_diff_mean_per_dim']:.4e}\n"
        f"Δ vars per dim: mean={compare['mean_abs_diff_var_per_dim']:.4e}, max={compare['max_abs_diff_var_per_dim']:.4e}\n"
        f"Δ cov eigs: mean={compare['mean_abs_diff_cov_eigs']:.4e}, max={compare['max_abs_diff_cov_eigs']:.4e}"
    )
    axs[1, 2].text(0.02, 0.98, text, va='top', ha='left', fontsize=10, family='monospace')
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{prefix}.png'), dpi=160)
    plt.close(fig)
    serializable = {
        'base': {k: v for k, v in base.items() if not isinstance(v, torch.Tensor)},
        'transformed': {k: v for k, v in transformed.items() if not isinstance(v, torch.Tensor)},
        'compare': compare,
    }
    with open(os.path.join(output_dir, f'{prefix}.json'), 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"[noise-report:{prefix}] saved png/json | base var={base['overall_var']:.6f}, transformed var={transformed['overall_var']:.6f}, mean|Δvar_d|={compare['mean_abs_diff_var_per_dim']:.6e}, mean|Δeig|={compare['mean_abs_diff_cov_eigs']:.6e}")
    return serializable


def save_noise_distribution_report(base_noise: torch.Tensor, transformed_noise: torch.Tensor, output_dir: str, patch_size: int, prefix: str, title: Optional[str] = None, label_a: str = 'base Gaussian', label_b: str = 'Q @ base Gaussian') -> Dict[str, object]:
    return save_patch_distribution_report(
        noise_to_patch_matrix(base_noise, patch_size),
        noise_to_patch_matrix(transformed_noise, patch_size),
        output_dir,
        prefix,
        title,
        label_a,
        label_b,
    )


def save_grid(images: torch.Tensor, path: str) -> None:
    from PIL import Image
    imgs = images.detach().cpu().float().clamp(0, 1)
    if imgs.ndim != 4:
        raise ValueError(f'Expected BCHW images, got shape={tuple(imgs.shape)}')
    b, c, h, w = imgs.shape
    if c == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    elif c > 3:
        imgs = imgs[:, :3]
    nrow = max(1, int(math.sqrt(b)))
    ncol = int(math.ceil(b / nrow))
    grid = torch.zeros(3, ncol * h, nrow * w, dtype=imgs.dtype)
    for idx in range(b):
        row = idx // nrow
        col = idx % nrow
        grid[:, row * h:(row + 1) * h, col * w:(col + 1) * w] = imgs[idx]
    array = (grid.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype('uint8')
    Image.fromarray(array).save(path)
