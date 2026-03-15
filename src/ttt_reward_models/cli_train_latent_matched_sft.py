import argparse

from .runners_sft import train_latent_matched_sft


def main() -> None:
    parser = argparse.ArgumentParser(description='Fine-tune a one-step SDXL Turbo model on assigned noises instead of random noises.')
    parser.add_argument('--manifest_path', type=str, required=True)
    parser.add_argument('--model_id', type=str, default='models/sdxl-turbo')
    parser.add_argument('--output_dir', type=str, default='outputs/latent_matched_sft')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--latent_loss_weight', type=float, default=1.0)
    parser.add_argument('--pixel_l1_weight', type=float, default=0.0)
    parser.add_argument('--preserve_latent_weight', type=float, default=0.0)
    parser.add_argument('--preserve_pixel_weight', type=float, default=0.0)
    parser.add_argument('--preserve_batch_size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--variant', type=str, default='fp16')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_every_epochs', type=int, default=1)
    args = parser.parse_args()

    train_latent_matched_sft(
        manifest_path=args.manifest_path,
        model_id=args.model_id,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        latent_loss_weight=args.latent_loss_weight,
        pixel_l1_weight=args.pixel_l1_weight,
        preserve_latent_weight=args.preserve_latent_weight,
        preserve_pixel_weight=args.preserve_pixel_weight,
        preserve_batch_size=args.preserve_batch_size,
        device=args.device,
        seed=args.seed,
        variant=args.variant,
        num_workers=args.num_workers,
        save_every_epochs=args.save_every_epochs,
    )


if __name__ == '__main__':
    main()
