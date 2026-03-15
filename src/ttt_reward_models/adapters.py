import torch
import torch.nn as nn


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
        # q = torch.matrix_exp(skew)

        # Use Cayley transform to get orthogonal matrix from skew-symmetric
        # Q = (I - S) @ inv(I + S)
        
        skew = self.raw - self.raw.transpose(0, 1)
        I = torch.eye(skew.size(0), device=skew.device, dtype=skew.dtype)
        q = torch.linalg.solve(I + skew, I - skew)
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
