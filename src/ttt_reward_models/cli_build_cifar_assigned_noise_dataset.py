import argparse

from .cifar_runners import build_cifar_assigned_noise_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='Assign matched latent noises for CIFAR-10 targets.')
    parser.add_argument('--generator_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/cifar_assigned_dataset')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--train_split', dest='train_split', action='store_true', default=True)
    parser.add_argument('--test_split', dest='train_split', action='store_false')
    parser.add_argument('--max_items', type=int, default=512)
    parser.add_argument('--assign_steps', type=int, default=300)
    parser.add_argument('--assign_lr', type=float, default=5e-2)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--mse_weight', type=float, default=1.0)
    parser.add_argument('--l1_weight', type=float, default=0.0)
    parser.add_argument('--prior_weight', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    build_cifar_assigned_noise_dataset(**vars(args))


if __name__ == '__main__':
    main()
