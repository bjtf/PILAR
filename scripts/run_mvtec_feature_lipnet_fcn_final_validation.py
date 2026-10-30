#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path

import torch


ARCHITECTURES = {"wide64": "wide64:64,64:3,3:2,2:128,64:true"}


def _print_command(command: list[str]) -> None:
    print("Command:", flush=True)
    print(" ".join(shlex.quote(x) for x in command), flush=True)


def _print_ranking(path: Path, top_k: int = 5) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    print(f"\nTop {min(top_k, len(rows))} macro rows from {path}:", flush=True)
    for row in rows[:top_k]:
        flops = row.get("approx_infer_flops_per_image", row.get("total_flops", "nan"))
        head_params = int(float(row["lipnet_params"]))
        backbone_params = int(float(row.get("backbone_params_frozen", 0) or 0))
        total_params = head_params + backbone_params
        print(
            "  "
            f"arch={row['architecture']} norm={row['score_norm']} "
            f"topk={float(row['image_topk_frac']):g} "
            f"image_auc={float(row['image_auc']):.4f} "
            f"image_ap={float(row['image_ap']):.4f} "
            f"pixel_auc={float(row['pixel_auc']):.4f} "
            f"aupro_30={float(row['aupro_30']):.4f} "
            f"head_params={head_params:,} "
            f"total_params={total_params:,} "
            f"flops={float(flops) / 1e9:.2f}G",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final all-class MVTec Feature-PILAR-FCN architecture validation "
            "with the paper configuration."
        )
    )
    parser.add_argument("--data_root", default="data/mvtec_ad")
    parser.add_argument("--categories", default="all")
    parser.add_argument("--device", default="mps_if_available")
    parser.add_argument("--output_dir", default="runs/benchmark/mvtec_feature_lipnet_fcn_final_validation_v1")
    parser.add_argument("--candidates", nargs="+", default=["wide64"], choices=sorted(ARCHITECTURES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_resize", type=int, default=512)
    parser.add_argument("--pro_eval_size", type=int, default=256)
    parser.add_argument("--fpr_limit", type=float, default=0.30)
    parser.add_argument("--num_thresholds", type=int, default=300)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if str(args.device) == "mps" and not torch.backends.mps.is_available():
        raise SystemExit(
            "Requested --device mps, but torch.backends.mps.is_available() is false in this Python environment. "
            "Use --device mps_if_available to fall back to CPU, or run from an environment where MPS is available."
        )

    architecture_specs = [ARCHITECTURES[name] for name in args.candidates]
    command = [
        args.python,
        str(Path(__file__).resolve().with_name("run_mvtec_feature_lipnet_fcn_arch_sweep.py")),
        "--data_root",
        str(args.data_root),
        "--categories",
        str(args.categories),
        "--device",
        str(args.device),
        "--output_dir",
        str(output_dir),
        "--architecture_preset",
        "none",
        "--architectures",
        *architecture_specs,
        "--feature_layers",
        "layer2,layer3",
        "--score_norms",
        "zscore",
        "--hard_loss_settings",
        "0.10:1.0",
        "--image_topk_fracs",
        "0.01",
        "--save_artifacts",
        "--target_metric",
        "image_auc",
        "--image_resize",
        str(args.image_resize),
        "--pro_eval_size",
        str(args.pro_eval_size),
        "--fpr_limit",
        str(args.fpr_limit),
        "--num_thresholds",
        str(args.num_thresholds),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--progress_every",
        str(args.progress_every),
    ]
    command.append("--resume" if args.resume else "--no-resume")

    _print_command(command)
    if args.dry_run:
        return

    subprocess.run(command, check=True)
    print(f"\nSaved summary: {output_dir / 'summary_arch_sweep.csv'}", flush=True)
    print(f"Saved ranking: {output_dir / 'ranking_arch_sweep.csv'}", flush=True)
    _print_ranking(output_dir / "ranking_arch_sweep.csv")


if __name__ == "__main__":
    main()
