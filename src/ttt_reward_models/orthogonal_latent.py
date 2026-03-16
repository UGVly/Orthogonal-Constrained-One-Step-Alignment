from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalLatentParameterization(nn.Module):
    """Parameterize z = Q eps with Q orthogonal via matrix exponential of a skew matrix."""

    def __init__(self, z_dim: int, base_noise: torch.Tensor, init_scale: float = 1e-4):
        super().__init__()
        if base_noise.ndim != 2 or base_noise.shape[0] != 1 or base_noise.shape[1] != z_dim:
            raise ValueError(f"Expected base_noise shape (1, {z_dim}), got {tuple(base_noise.shape)}")
        self.z_dim = z_dim
        self.register_buffer('base_noise', base_noise.detach().clone())
        raw = torch.randn(z_dim, z_dim, device=base_noise.device, dtype=base_noise.dtype) * init_scale
        self.raw = nn.Parameter(raw)

    def orthogonal_matrix(self) -> torch.Tensor:
        skew = self.raw - self.raw.transpose(0, 1)
        return torch.matrix_exp(skew)

    def forward(self) -> torch.Tensor:
        q = self.orthogonal_matrix()
        return self.base_noise @ q.transpose(0, 1)


@dataclass
class OrthogonalAssignmentResult:
    best_noise: torch.Tensor
    best_base_noise: torch.Tensor
    best_q: torch.Tensor
    best_loss: float
    curve: List[float]
    metrics: Dict[str, float]
    best_recon: Optional[torch.Tensor] = None


@torch.no_grad()
def summarize_orthogonal_map(q: torch.Tensor) -> Dict[str, float]:
    q_cpu = q.detach().cpu().to(torch.float64)
    eye = torch.eye(q_cpu.shape[0], dtype=q_cpu.dtype)
    s = torch.linalg.svdvals(q_cpu)
    return {
        'orthogonality_fro': float(torch.linalg.matrix_norm(q_cpu.transpose(0, 1) @ q_cpu - eye).item()),
        'sv_min': float(s.min().item()),
        'sv_max': float(s.max().item()),
        'sv_mean': float(s.mean().item()),
        'det_abs': float(torch.linalg.det(q_cpu).abs().item()),
    }


@torch.no_grad()
def summarize_noise_moments(noise: torch.Tensor) -> Dict[str, float]:
    flat = noise.detach().cpu().float().reshape(-1)
    return {
        'noise_mean': float(flat.mean().item()),
        'noise_std': float(flat.std(unbiased=True).item()) if flat.numel() > 1 else 0.0,
        'noise_var': float(flat.var(unbiased=True).item()) if flat.numel() > 1 else 0.0,
        'noise_l2': float(torch.linalg.vector_norm(flat).item()),
    }


def optimize_orthogonal_noise_for_target(
    *,
    generator: nn.Module,
    target: torch.Tensor,
    label: torch.Tensor,
    z_dim: int,
    assign_steps: int,
    assign_lr: float,
    mse_weight: float = 1.0,
    l1_weight: float = 0.0,
    prior_weight: float = 1e-3,
    orth_reg_weight: float = 0.0,
    init_scale: float = 1e-4,
) -> OrthogonalAssignmentResult:
    device = target.device
    dtype = target.dtype
    base_noise = torch.randn(1, z_dim, device=device, dtype=dtype)
    param = OrthogonalLatentParameterization(z_dim=z_dim, base_noise=base_noise, init_scale=init_scale).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(param.parameters(), lr=assign_lr)

    best_loss = float('inf')
    best_noise = None
    best_base_noise = None
    best_q = None
    best_recon = None
    curve: List[float] = []

    was_training = generator.training
    generator.eval()
    try:
        for _ in range(assign_steps):
            optimizer.zero_grad(set_to_none=True)
            z = param()
            recon = generator(z, label)
            mse = F.mse_loss(recon, target)
            l1 = F.l1_loss(recon, target) if l1_weight > 0 else torch.zeros_like(mse)
            q = param.orthogonal_matrix()
            prior = (z.pow(2).mean() - 1.0).pow(2) + z.mean().pow(2)
            ortho_penalty = torch.linalg.matrix_norm(q.transpose(0, 1) @ q - torch.eye(z_dim, device=device, dtype=dtype)).pow(2)
            loss = mse_weight * mse + l1_weight * l1 + prior_weight * prior + orth_reg_weight * ortho_penalty
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            curve.append(loss_value)
            if loss_value < best_loss:
                best_loss = loss_value
                best_noise = z.detach().cpu().clone()
                best_base_noise = param.base_noise.detach().cpu().clone()
                best_q = q.detach().cpu().clone()
                best_recon = recon.detach().cpu().clone()
    finally:
        generator.train(was_training)

    if best_noise is None or best_base_noise is None or best_q is None:
        raise RuntimeError('Orthogonal latent assignment failed to produce a result.')

    metrics = summarize_orthogonal_map(best_q)
    metrics.update(summarize_noise_moments(best_noise))
    metrics.update({
        'base_noise_mean': float(best_base_noise.float().mean().item()),
        'base_noise_var': float(best_base_noise.float().var(unbiased=True).item()) if best_base_noise.numel() > 1 else 0.0,
        'base_noise_std': float(best_base_noise.float().std(unbiased=True).item()) if best_base_noise.numel() > 1 else 0.0,
        'norm_diff_abs': float(abs(torch.linalg.vector_norm(best_noise.float()) - torch.linalg.vector_norm(best_base_noise.float())).item()),
    })

    return OrthogonalAssignmentResult(
        best_noise=best_noise,
        best_base_noise=best_base_noise,
        best_q=best_q,
        best_loss=best_loss,
        curve=curve,
        metrics=metrics,
        best_recon=best_recon,
    )
