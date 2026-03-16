from __future__ import annotations

import torch
import torch.nn as nn


class CIFARConditionalGenerator(nn.Module):
    def __init__(self, z_dim: int = 128, num_classes: int = 10, embed_dim: int = 64, base_channels: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(z_dim + embed_dim, base_channels * 8 * 4 * 4),
            nn.BatchNorm1d(base_channels * 8 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        y = self.label_emb(labels)
        x = torch.cat([z, y], dim=1)
        x = self.fc(x)
        x = x.view(x.shape[0], -1, 4, 4)
        return self.net(x)


class CIFARConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes: int = 10, base_channels: int = 64, feat_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(base_channels * 4 * 4 * 4, feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(feat_dim, 1)
        self.label_emb = nn.Embedding(num_classes, feat_dim)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feat = self.features(images)
        logits = self.fc(feat).squeeze(1)
        proj = (feat * self.label_emb(labels)).sum(dim=1)
        return logits + proj


@torch.no_grad()
def sample_cifar_grid(
    generator: CIFARConditionalGenerator,
    *,
    device: str,
    z: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    per_class: int = 8,
) -> torch.Tensor:
    generator.eval()
    if labels is None:
        labels = torch.arange(generator.num_classes, device=device).repeat_interleave(per_class)
    if z is None:
        z = torch.randn(labels.shape[0], generator.z_dim, device=device)
    out = generator(z, labels).detach()
    generator.train()
    return (out + 1.0) * 0.5
