import argparse

from .runners_sft import build_assigned_noise_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a latent-matched SFT dataset from pairs.jsonl by assigning the best noise to each sample.')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--model_id', type=str, default='models/sdxl-turbo')
    parser.add_argument('--image_field', type=str, default='image')
    parser.add_argument('--prompt_field', type=str, default='prompt')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--no_link_image', action='store_true')
    parser.add_argument('--no_skip_existing', action='store_true')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--latent_loss_weight', type=float, default=1.0)
    parser.add_argument('--pixel_l1_weight', type=float, default=0.0)
    parser.add_argument('--drift_reg_weight', type=float, default=0.0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--variant', type=str, default='fp16')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--print_eigs_every', type=int, default=1)
    args = parser.parse_args()

    build_assigned_noise_dataset(
        data_root=args.data_root,
        output_root=args.output_root,
        model_id=args.model_id,
        image_field=args.image_field,
        prompt_field=args.prompt_field,
        start_index=args.start_index,
        max_samples=args.max_samples,
        link_image=not args.no_link_image,
        skip_existing=not args.no_skip_existing,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patch_size=args.patch_size,
        latent_loss_weight=args.latent_loss_weight,
        pixel_l1_weight=args.pixel_l1_weight,
        drift_reg_weight=args.drift_reg_weight,
        device=args.device,
        seed=args.seed,
        variant=args.variant,
        save_every=args.save_every,
        print_eigs_every=args.print_eigs_every,
    )


if __name__ == '__main__':
    main()
