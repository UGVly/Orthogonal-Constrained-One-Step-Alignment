import argparse

from .mnist_runners import run_mnist_em_latent_matched_sft


def main() -> None:
    parser = argparse.ArgumentParser(description='EM-style latent-matched SFT on MNIST.')
    parser.add_argument('--generator_ckpt', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='outputs/mnist_em_latent_matched_sft')
    parser.add_argument('--em_rounds', type=int, default=3)
    parser.add_argument('--assign_max_items', type=int, default=256)
    parser.add_argument('--assign_steps', type=int, default=200)
    parser.add_argument('--assign_lr', type=float, default=5e-2)
    parser.add_argument('--sft_epochs', type=int, default=2)
    parser.add_argument('--sft_batch_size', type=int, default=64)
    parser.add_argument('--sft_lr', type=float, default=1e-4)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_mnist_em_latent_matched_sft(**vars(args))


if __name__ == '__main__':
    main()
