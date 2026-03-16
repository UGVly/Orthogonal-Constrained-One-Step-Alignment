import argparse

from .cifar_runners import train_cifar_one_step_gan


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a one-step conditional GAN on CIFAR-10.')
    parser.add_argument('--output_dir', type=str, default='outputs/cifar_gan')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--generator_lr', type=float, default=2e-4)
    parser.add_argument('--discriminator_lr', type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--train_limit', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train_cifar_one_step_gan(**vars(args))


if __name__ == '__main__':
    main()
