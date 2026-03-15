import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterator, Optional

from tqdm import tqdm

from src.test_time_inference.test_time_oft_noise_sdxl_turbo_sft import run_test_time_oft_sft


def iter_pairs(
    pairs_jsonl: Path,
    image_field: str,
    start_index: int = 0,
    max_samples: Optional[int] = None,
) -> Iterator[Dict[str, str]]:
    yielded = 0
    with pairs_jsonl.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx < start_index:
                continue
            if max_samples is not None and yielded >= max_samples:
                break

            item = json.loads(line)
            if image_field not in item:
                raise KeyError(f"Field '{image_field}' not found in line {line_idx}: {item.keys()}")
            item["_line_idx"] = line_idx
            yielded += 1
            yield item


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_or_link_image(src: Path, dst: Path, link_image: bool) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_image:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def build_dataset(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).resolve()
    pairs_jsonl = data_root / "pairs.jsonl"
    if not pairs_jsonl.exists():
        raise FileNotFoundError(f"pairs.jsonl not found: {pairs_jsonl}")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    samples_root = output_root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifest.jsonl"
    failures_path = output_root / "failures.jsonl"

    pair_iter = iter_pairs(
        pairs_jsonl=pairs_jsonl,
        image_field=args.image_field,
        start_index=args.start_index,
        max_samples=args.max_samples,
    )

    success_count = 0
    fail_count = 0

    for item in tqdm(pair_iter, desc="OFT-SFT dataset build"):
        line_idx = int(item["_line_idx"])
        prompt = str(item["prompt"])
        image_rel = str(item[args.image_field])
        image_abs = (data_root / image_rel).resolve()
        if not image_abs.exists():
            raise FileNotFoundError(f"Image not found for line {line_idx}: {image_abs}")

        image_stem = Path(image_rel).stem
        sample_id = f"{line_idx:06d}_{image_stem}"
        sample_dir = samples_root / sample_id
        run_dir = sample_dir / "run_outputs"
        meta_path = sample_dir / "meta.json"

        if args.skip_existing and meta_path.exists():
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)

        try:
            _, stats = run_test_time_oft_sft(
                prompt,
                target_image_path=str(image_abs),
                model_id=args.model_id,
                output_dir=str(run_dir),
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
                seed=args.seed + line_idx,
                variant=args.variant,
                save_every=args.save_every,
                print_eigs_every=args.print_eigs_every,
                save_outputs=True,
            )

            best_noise_src = run_dir / "best_input_noise.pt"
            if not best_noise_src.exists():
                best_noise_src = run_dir / "best_noise.pt"
            if not best_noise_src.exists():
                raise FileNotFoundError(f"best input noise file missing in {run_dir}")

            ref_dst = sample_dir / f"reference{image_abs.suffix.lower()}"
            noise_dst = sample_dir / "best_input_noise.pt"
            copy_or_link_image(image_abs, ref_dst, link_image=args.link_image)
            shutil.copy2(best_noise_src, noise_dst)

            record = {
                "sample_id": sample_id,
                "prompt": prompt,
                "source": {
                    "line_idx": line_idx,
                    "image_field": args.image_field,
                    "image_rel": image_rel,
                    "image_abs": str(image_abs),
                },
                "artifacts": {
                    "reference_image": str(ref_dst.relative_to(output_root)),
                    "best_input_noise": str(noise_dst.relative_to(output_root)),
                    "run_outputs_dir": str(run_dir.relative_to(output_root)),
                },
                "stats": stats,
            }

            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            success_count += 1
        except Exception as exc:
            fail_count += 1
            fail_record = {
                "sample_id": sample_id,
                "line_idx": line_idx,
                "prompt": prompt,
                "image_rel": image_rel,
                "error": repr(exc),
            }
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(fail_record, ensure_ascii=False) + "\n")
            if args.fail_fast:
                raise

    print(f"Done. success={success_count}, failed={fail_count}, output={output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run OFT noise SFT and export a dataset of {prompt, reference image, best input noise}."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/storage/wenyandongLab/jiangzhou/CODE/OneStepAlign/datasets/pickapicv2_export_5000",
        help="Dataset root containing pairs.jsonl and images/.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/storage/wenyandongLab/jiangzhou/CODE/OneStepAlign/datasets/pickapicv2_oft_noise_sft",
        help="Output dataset root directory.",
    )
    parser.add_argument(
        "--image_field",
        type=str,
        default="chosen",
        choices=["chosen", "rejected"],
        help="Use which image path field from pairs.jsonl.",
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--link_image", action="store_true", help="Symlink reference image instead of copying.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop on first failure.")

    parser.add_argument("--model_id", type=str, default="./models/sdxl-turbo")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--latent_loss_weight", type=float, default=1.0)
    parser.add_argument("--pixel_l1_weight", type=float, default=0.1)
    parser.add_argument("--drift_reg_weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", type=str, default="fp16")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--print_eigs_every", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
