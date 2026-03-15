import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from test_time_oft_noise_sdxl_turbo_pickscore import (
    run_test_time_oft_pickscore_for_prompts,
)


def read_prompts(path: str) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images from a prompt list using OneStep OFT + PickScore.")
    parser.add_argument("--prompt_file", type=str, required=True, help="一行一个 prompt 的文本文件")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录，每个 prompt 存到 prompt_{i}")
    parser.add_argument(
        "--model_id",
        type=str,
        default="/home/jiangzhou/.cache/modelscope/hub/models/AI-ModelScope/sdxl-turbo/",
    )
    parser.add_argument("--pickscore_model_id", type=str, default="yuvalkirstain/PickScore_v1")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", type=str, default="fp16")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--print_eigs_every", type=int, default=1)
    args = parser.parse_args()

    prompts = read_prompts(args.prompt_file)
    if not prompts:
        raise RuntimeError(f"Prompt file is empty: {args.prompt_file}")

    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    results: Dict[str, Tuple[List[torch.Tensor], Dict[str, object]]] = run_test_time_oft_pickscore_for_prompts(
        prompts,
        model_id=args.model_id,
        pickscore_model_id=args.pickscore_model_id,
        output_dir=args.output_dir,
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

    records = []
    for idx, prompt in enumerate(prompts):
        _, reward_dict = results[prompt]
        reward_history = [float(x) for x in reward_dict.get("reward_history", [])]
        pickscore_history = [float(x) for x in reward_dict.get("pickscore_history", [])]
        records.append(
            {
                "index": idx,
                "prompt": prompt,
                # Backward-compatible fields (best metrics).
                "reward": float(reward_dict.get("reward", float("nan"))),
                "pickscore": float(reward_dict.get("pickscore", float("nan"))),
                # Explicit best/last metrics.
                "best_reward": float(reward_dict.get("best_reward", reward_dict.get("reward", float("nan")))),
                "best_pickscore": float(reward_dict.get("best_pickscore", reward_dict.get("pickscore", float("nan")))),
                "last_reward": float(reward_dict.get("last_reward", float("nan"))),
                "last_pickscore": float(reward_dict.get("last_pickscore", float("nan"))),
                # Each-step trajectories.
                "reward_history": reward_history,
                "pickscore_history": pickscore_history,
                "num_steps": len(reward_history),
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics_per_prompt.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved per-prompt metrics to: {metrics_path}")

    valid = [r for r in records if not torch.isnan(torch.tensor(float(r["best_reward"])))]
    if valid:
        n = len(valid)
        max_steps = max(int(r.get("num_steps", 0)) for r in valid)
        mean_reward_per_step = []
        mean_pickscore_per_step = []
        for step_idx in range(max_steps):
            step_rewards = []
            step_picks = []
            for r in valid:
                rh = r.get("reward_history", [])
                ph = r.get("pickscore_history", [])
                if step_idx < len(rh):
                    v = float(rh[step_idx])
                    if math.isfinite(v):
                        step_rewards.append(v)
                if step_idx < len(ph):
                    v = float(ph[step_idx])
                    if math.isfinite(v):
                        step_picks.append(v)
            mean_reward_per_step.append(sum(step_rewards) / len(step_rewards) if step_rewards else float("nan"))
            mean_pickscore_per_step.append(sum(step_picks) / len(step_picks) if step_picks else float("nan"))

        summary = {
            "num_samples": n,
            "mean_best_reward": sum(float(r["best_reward"]) for r in valid) / n,
            "mean_best_pickscore": sum(float(r["best_pickscore"]) for r in valid) / n,
            "mean_last_reward": sum(float(r["last_reward"]) for r in valid) / n,
            "mean_last_pickscore": sum(float(r["last_pickscore"]) for r in valid) / n,
            "mean_reward_per_step": mean_reward_per_step,
            "mean_pickscore_per_step": mean_pickscore_per_step,
        }
        summary_path = out_dir / "metrics_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("Summary:", json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()

