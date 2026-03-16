import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--model_id', type=str, default='models/sdxl-turbo')
    parser.add_argument('--output_dir', type=str, default='outputs/test_time_oft_imagereward')
    parser.add_argument('--imagereward_root', type=str, default=None)
    parser.add_argument('--imagereward_model_path', type=str, default=None)
    parser.add_argument('--imagereward_med_config_path', type=str, default=None)
    parser.add_argument('--imagereward_auto_download', action='store_true')
    parser.add_argument('--no_prefer_modelscope', action='store_true')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--variant', type=str, default='fp16')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--print_eigs_every', type=int, default=1)
    args = parser.parse_args()
    from .runners import run_test_time_oft_imagereward

    run_test_time_oft_imagereward(
        args.prompt,
        model_id=args.model_id,
        output_dir=args.output_dir,
        imagereward_root=args.imagereward_root,
        imagereward_model_path=args.imagereward_model_path,
        imagereward_med_config_path=args.imagereward_med_config_path,
        imagereward_auto_download=args.imagereward_auto_download,
        prefer_modelscope=not args.no_prefer_modelscope,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patch_size=args.patch_size,
        device=args.device,
        seed=args.seed,
        variant=args.variant,
        save_every=args.save_every,
        print_eigs_every=args.print_eigs_every,
        save_outputs=True,
    )


if __name__ == '__main__':
    main()
