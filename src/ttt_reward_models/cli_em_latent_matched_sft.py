import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description='Alternating latent assignment + one-step SFT.')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--work_dir', type=str, required=True)
    parser.add_argument('--initial_model_id', type=str, default='models/sdxl-turbo')
    parser.add_argument('--num_cycles', type=int, default=2)
    parser.add_argument('--image_field', type=str, default='image')
    parser.add_argument('--prompt_field', type=str, default='prompt')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--assign_steps', type=int, default=30)
    parser.add_argument('--assign_lr', type=float, default=5e-4)
    parser.add_argument('--train_epochs', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--train_lr', type=float, default=1e-5)
    parser.add_argument('--latent_loss_weight', type=float, default=1.0)
    parser.add_argument('--pixel_l1_weight', type=float, default=0.0)
    parser.add_argument('--preserve_latent_weight', type=float, default=0.0)
    parser.add_argument('--preserve_pixel_weight', type=float, default=0.0)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--variant', type=str, default='fp16')
    args = parser.parse_args()
    from .runners_sft import run_em_latent_matched_sft

    run_em_latent_matched_sft(
        data_root=args.data_root,
        work_dir=args.work_dir,
        initial_model_id=args.initial_model_id,
        num_cycles=args.num_cycles,
        image_field=args.image_field,
        prompt_field=args.prompt_field,
        height=args.height,
        width=args.width,
        assign_steps=args.assign_steps,
        assign_lr=args.assign_lr,
        train_epochs=args.train_epochs,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        latent_loss_weight=args.latent_loss_weight,
        pixel_l1_weight=args.pixel_l1_weight,
        preserve_latent_weight=args.preserve_latent_weight,
        preserve_pixel_weight=args.preserve_pixel_weight,
        patch_size=args.patch_size,
        device=args.device,
        seed=args.seed,
        variant=args.variant,
    )


if __name__ == '__main__':
    main()
