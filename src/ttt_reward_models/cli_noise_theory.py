import argparse
import json
import os

import torch

from .adapters import PatchOrthogonalNoise
from .diagnostics import save_patch_distribution_report
from .utils import ensure_dir, get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description='Standalone verification that orthogonal transforms preserve standard Gaussian statistics.')
    parser.add_argument('--output_dir', type=str, default='outputs/orthogonal_gaussian_theory')
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=65536)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--init_scale', type=float, default=1e-1)
    parser.add_argument('--adapter_ckpt', type=str, default=None)
    args = parser.parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.seed)
    device = get_device(args.device)
    adapter = PatchOrthogonalNoise(channels=args.channels, patch_size=args.patch_size, init_scale=args.init_scale).to(device)
    if args.adapter_ckpt is not None:
        state = torch.load(args.adapter_ckpt, map_location='cpu')
        adapter.load_state_dict(state)
    with torch.no_grad():
        q = adapter.orthogonal_matrix().detach().to(device=device, dtype=torch.float32)
    block_dim = args.channels * args.patch_size * args.patch_size
    chunks = []
    transformed_chunks = []
    remaining = args.num_samples
    while remaining > 0:
        cur = min(remaining, args.batch_size)
        z = torch.randn(cur, block_dim, device=device, dtype=torch.float32)
        qz = z @ q.T
        chunks.append(z.cpu())
        transformed_chunks.append(qz.cpu())
        remaining -= cur
    z_all = torch.cat(chunks, dim=0)
    qz_all = torch.cat(transformed_chunks, dim=0)
    save_patch_distribution_report(
        z_all,
        qz_all,
        args.output_dir,
        'orthogonal_gaussian_verification',
        title='Orthogonal transform preserves a standard Gaussian',
        label_a='z ~ N(0, I)',
        label_b='Qz, Q orthogonal',
    )
    metadata = {
        'channels': args.channels,
        'patch_size': args.patch_size,
        'block_dim': block_dim,
        'num_samples': args.num_samples,
        'seed': args.seed,
        'device': device,
        'adapter_ckpt': args.adapter_ckpt,
        'q_shape': list(q.shape),
        'q_orthogonality_error_fro': float(torch.linalg.matrix_norm(q.cpu().T @ q.cpu() - torch.eye(q.shape[0])).item()),
    }
    with open(os.path.join(args.output_dir, 'orthogonal_gaussian_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print('Saved standalone orthogonal-Gaussian verification to:', args.output_dir)
    print('Main figure:', os.path.join(args.output_dir, 'orthogonal_gaussian_verification.png'))
    print('Summary json:', os.path.join(args.output_dir, 'orthogonal_gaussian_verification.json'))


if __name__ == '__main__':
    main()
