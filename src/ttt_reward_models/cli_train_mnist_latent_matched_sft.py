import argparse

from .mnist_runners import train_mnist_latent_matched_sft


def main() -> None:
    parser = argparse.ArgumentParser(description='Latent-matched SFT for a one-step MNIST generator.')
    parser.add_argument('--generator_ckpt', type=str, required=True)
    parser.add_argument('--manifest_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/mnist_latent_matched_sft')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--mse_weight', type=float, default=1.0)
    parser.add_argument('--l1_weight', type=float, default=0.0)
    parser.add_argument('--preserve_weight', type=float, default=0.25)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train_mnist_latent_matched_sft(**vars(args))


if __name__ == '__main__':
    main()
