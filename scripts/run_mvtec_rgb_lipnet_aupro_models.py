#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


CONFIGS = {
    "whole_rgb": {
        "output_dir": "runs/benchmark/mvtec_rgb_lipnet_whole_resize512_aupro30_v1",
        "args": [
            "--image_resize",
            "512",
            "--patch_size",
            "512",
            "--train_patch_stride",
            "512",
            "--test_patch_stride",
            "512",
            "--max_train_patches",
            "8000",
            "--batch_size",
            "8",
            "--eval_pixel_auc",
            "--eval_aupro",
            "--pixel_heatmap_mode",
            "residual",
        ],
    },
    "patch_rgb": {
        "output_dir": "runs/benchmark/mvtec_patch_lipnet_rgb_all_resize512_s128_aupro30_v1",
        "args": [
            "--image_resize",
            "512",
            "--patch_size",
            "128",
            "--train_patch_stride",
            "64",
            "--test_patch_stride",
            "64",
            "--max_train_patches",
            "8000",
            "--batch_size",
            "128",
            "--eval_pixel_auc",
            "--eval_aupro",
            "--pixel_heatmap_mode",
            "residual",
        ],
    },
}


def _parse_candidates(value: str) -> list[str]:
    out = [x.strip() for x in str(value).split(",") if x.strip()]
    unknown = [x for x in out if x not in CONFIGS]
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}. Available: {sorted(CONFIGS)}")
    if not out or len(set(out)) != len(out):
        raise ValueError("Candidates must be nonempty and unique")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MVTec RGB PILAR AUPRO@30 validation for the two RGB-only models.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="runs/mvtec_rgb_controls")
    parser.add_argument("--device", default="mps_if_available")
    parser.add_argument("--categories", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--candidates", default="whole_rgb,patch_rgb")
    parser.add_argument("--fpr_limit", type=float, default=0.30)
    parser.add_argument("--num_thresholds", type=int, default=400)
    parser.add_argument("--pro_eval_size", type=int, default=256)
    parser.add_argument("--pixel_auc_size", type=int, default=256)
    parser.add_argument("--whole_batch_size", type=int, default=8)
    parser.add_argument("--patch_batch_size", type=int, default=128)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    candidates = _parse_candidates(args.candidates)
    script = Path(__file__).with_name("run_mvtec_patch_lipnet.py")
    for candidate in candidates:
        cfg = CONFIGS[candidate]
        output_dir = str(Path(args.output_dir) / candidate)
        batch_size = int(args.whole_batch_size) if candidate == "whole_rgb" else int(args.patch_batch_size)
        cfg_args = list(cfg["args"])
        batch_idx = cfg_args.index("--batch_size") + 1
        cfg_args[batch_idx] = str(batch_size)
        cmd = [
            sys.executable,
            str(script),
            "--data_root",
            str(args.data_root),
            "--categories",
            str(args.categories),
            "--device",
            str(args.device),
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--num_workers",
            str(args.num_workers),
            "--output_dir",
            output_dir,
            "--fpr_limit",
            str(args.fpr_limit),
            "--num_thresholds",
            str(args.num_thresholds),
            "--pro_eval_size",
            str(args.pro_eval_size),
            "--pixel_auc_size",
            str(args.pixel_auc_size),
            "--progress_every",
            str(args.progress_every),
            *cfg_args,
        ]
        print(f"\n[{candidate}] {shlex.join(cmd)}", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
            print(f"[{candidate}] saved {output_dir}/summary_aupro.csv", flush=True)


if __name__ == "__main__":
    main()
