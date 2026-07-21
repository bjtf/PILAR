#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from lipnet.efficiency import estimate_forward_flops_per_sample, model_size_metrics
from lipnet.utils import resolve_device, set_seed
from run_mvtec_feature_lipnet_fcn_aupro import _count_params, _evaluate_norm, _finite_mean
from run_mvtec_feature_lipnet_fcn_sweep import (
    MVTecImageDataset,
    ResNetMultiFeatureExtractor,
    _build_models,
    _collect_train_maps,
    _parse_hard_setting,
    _parse_layer_set,
    _safe_label,
    _train_models,
)
from run_mvtec_patch_lipnet import _category_names, _find_mvtec_root, _list_images, _test_images, _write_csv


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    encoder_channels: str
    rf_schedule: str
    overlap_schedule: str
    decoder_channels: str
    refine_decoder: bool


def _parse_bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def _parse_architecture(value: str) -> ArchitectureSpec:
    parts = [x.strip() for x in str(value).split(":")]
    if len(parts) != 6:
        raise ValueError(
            "Architecture specs must be name:encoder_channels:rf_schedule:overlap_schedule:decoder_channels:refine, "
            f"got {value!r}"
        )
    return ArchitectureSpec(
        name=parts[0],
        encoder_channels=parts[1],
        rf_schedule=parts[2],
        overlap_schedule=parts[3],
        decoder_channels=parts[4],
        refine_decoder=_parse_bool(parts[5]),
    )


def _preset_architectures(preset: str) -> list[ArchitectureSpec]:
    if preset == "none":
        return []
    quick = [
        ArchitectureSpec("base32", "32,32", "3,3", "2,2", "64,32", True),
        ArchitectureSpec("wide48", "48,48", "3,3", "2,2", "96,48", True),
        ArchitectureSpec("wide64", "64,64", "3,3", "2,2", "128,64", True),
        ArchitectureSpec("taper64_32", "64,32", "3,3", "2,2", "128,64", True),
        ArchitectureSpec("deep32x3", "32,32,32", "3,3,3", "2,2,2", "64,32", True),
    ]
    if preset == "quick":
        return quick
    if preset == "full":
        return quick + [
            ArchitectureSpec("small24", "24,24", "3,3", "2,2", "48,24", True),
            ArchitectureSpec("base32_dec128", "32,32", "3,3", "2,2", "128,64", True),
            ArchitectureSpec("base32_dec128x3", "32,32", "3,3", "2,2", "128,64,32", True),
            ArchitectureSpec("rf5_32", "32,32", "5,3", "4,2", "64,32", True),
            ArchitectureSpec("no_refine32", "32,32", "3,3", "2,2", "64,32", False),
        ]
    raise ValueError("--architecture_preset must be none, quick, or full")


def _clone_args(args: argparse.Namespace, **updates: object) -> argparse.Namespace:
    payload = dict(vars(args))
    payload.update(updates)
    return argparse.Namespace(**payload)


def _label_float(value: float) -> str:
    return _safe_label(f"{float(value):g}")


def _architecture_label(arch: ArchitectureSpec) -> str:
    return _safe_label(arch.name)


def _completed_marker(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _read_single_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def _is_architecture_size_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "kernel_size" in text and ("spatial size" in text or "sliding blocks" in text)


def _run_category_architecture(
    category: str,
    mvtec_root: Path,
    arch: ArchitectureSpec,
    hard_setting: tuple[float, float],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict]:
    category_root = mvtec_root / category
    train_paths = _list_images(category_root / "train" / "good")
    test_rows = _test_images(category_root)
    if int(args.max_train_images) > 0:
        train_paths = train_paths[: int(args.max_train_images)]
    if int(args.max_test_images) > 0:
        test_rows = test_rows[: int(args.max_test_images)]
    if not train_paths:
        raise RuntimeError(f"{category}: no train/good images found")
    if not test_rows:
        raise RuntimeError(f"{category}: no test images found")

    layers = _parse_layer_set(args.feature_layers)
    hard_frac, hard_weight = hard_setting
    hard_label = f"{hard_frac:g}:{hard_weight:g}"
    arch_label = _architecture_label(arch)
    hard_dir = f"hard_{_safe_label(hard_label)}"

    arch_args = _clone_args(
        args,
        feature_encoder_channels=arch.encoder_channels,
        feature_rf_schedule=arch.rf_schedule,
        feature_overlap_schedule=arch.overlap_schedule,
        feature_decoder_channels=arch.decoder_channels,
        refine_decoder=arch.refine_decoder,
    )

    train_loader = DataLoader(
        MVTecImageDataset(train_paths, image_resize=int(args.image_resize)),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=False,
    )
    sample = next(iter(train_loader))[:1]
    backbone = ResNetMultiFeatureExtractor(args.backbone, layers, args.backbone_weights).to(device).eval()
    models, shapes = _build_models(backbone, layers, sample, device, arch_args)

    print(
        f"{category}: arch={arch.name} layers={','.join(layers)} hard={hard_label} "
        f"train_images={len(train_paths)} test_images={len(test_rows)} shapes={shapes}",
        flush=True,
    )
    try:
        _train_models(models, backbone, train_loader, device, arch_args, hard_frac, hard_weight)
        train_maps = _collect_train_maps(models, backbone, train_loader, device, arch_args)
    except RuntimeError as exc:
        if _is_architecture_size_error(exc):
            print(f"{category}: skipping incompatible architecture {arch.name}: {exc}", flush=True)
            return []
        raise

    backbone_flops = estimate_forward_flops_per_sample(backbone, (3, int(args.image_resize), int(args.image_resize)))
    backbone_params = _count_params(backbone, trainable_only=False)
    lipnet_params = int(sum(model_size_metrics(model)["param_count"] for model in models.values()))
    lipnet_bytes = int(sum(model_size_metrics(model)["param_bytes"] for model in models.values()))
    lipnet_flops = float(
        sum(estimate_forward_flops_per_sample(model, shapes[layer])["flops_forward_total"] for layer, model in models.items())
    )

    rows = []
    for topk_frac in [float(x) for x in args.image_topk_fracs]:
        topk_dir = f"topk_{_label_float(topk_frac)}"
        for score_norm in args.score_norms:
            eval_dir = Path(args.output_dir) / "details" / arch_label / hard_dir / topk_dir
            row_path = eval_dir / f"summary_row_{category}_{score_norm}.csv"
            if bool(args.resume) and _completed_marker(row_path):
                cached = _read_single_row(row_path)
                if cached is not None:
                    rows.append(cached)
                print(f" {category}/{arch.name}/{hard_label}/{score_norm}/topk={topk_frac:g}: loaded existing result")
                continue

            eval_args = _clone_args(
                arch_args,
                image_topk_frac=topk_frac,
                output_dir=eval_dir,
            )
            row = _evaluate_norm(
                category,
                category_root,
                test_rows,
                models,
                backbone,
                train_maps,
                ",".join(layers),
                hard_label,
                str(score_norm),
                device,
                eval_args,
            )
            row.update(
                {
                    "architecture": arch.name,
                    "feature_encoder_channels": arch.encoder_channels,
                    "feature_rf_schedule": arch.rf_schedule,
                    "feature_overlap_schedule": arch.overlap_schedule,
                    "feature_decoder_channels": arch.decoder_channels,
                    "refine_decoder": int(arch.refine_decoder),
                    "image_topk_frac": float(topk_frac),
                    "n_train_images": int(len(train_paths)),
                    "image_resize": int(args.image_resize),
                    "backbone": str(args.backbone),
                    "backbone_params_frozen": int(backbone_params),
                    "lipnet_params": int(lipnet_params),
                    "lipnet_param_bytes": int(lipnet_bytes),
                    "feature_shapes": ";".join(f"{k}:{'x'.join(map(str, shapes[k]))}" for k in layers),
                    "backbone_flops_per_image": float(backbone_flops["flops_forward_total"]),
                    "lipnet_flops_per_image": float(lipnet_flops),
                    "approx_infer_flops_per_image": float(backbone_flops["flops_forward_total"] + lipnet_flops),
                    "train_topk_feature_frac": float(hard_frac),
                    "train_topk_feature_weight": float(hard_weight),
                    "quantile_low": float(args.quantile_low),
                    "quantile_high": float(args.quantile_high),
                }
            )
            _write_csv(row_path, [row], _fieldnames())
            rows.append(row)
    return rows


def _fieldnames() -> list[str]:
    return [
        "category",
        "architecture",
        "feature_layers",
        "hard_loss",
        "score_norm",
        "image_topk_frac",
        "image_auc",
        "image_ap",
        "pixel_auc",
        "aupro_30",
        "aupro_30_percent",
        "fpr_limit",
        "num_thresholds",
        "n_pro_regions",
        "n_normal_pixels",
        "n_train_images",
        "n_test_images",
        "n_test_anomaly",
        "image_resize",
        "pro_eval_size",
        "connectivity",
        "backbone",
        "backbone_params_frozen",
        "lipnet_params",
        "lipnet_param_bytes",
        "feature_shapes",
        "backbone_flops_per_image",
        "lipnet_flops_per_image",
        "approx_infer_flops_per_image",
        "feature_encoder_channels",
        "feature_rf_schedule",
        "feature_overlap_schedule",
        "feature_decoder_channels",
        "refine_decoder",
        "train_topk_feature_frac",
        "train_topk_feature_weight",
        "quantile_low",
        "quantile_high",
    ]


def _group_key(row: dict) -> tuple:
    return (
        row["architecture"],
        row["feature_layers"],
        row["hard_loss"],
        row["score_norm"],
        float(row["image_topk_frac"]),
    )


def _macro_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    out = []
    for _, group in sorted(grouped.items(), key=lambda item: item[0]):
        first = group[0]
        macro = {name: "" for name in _fieldnames()}
        for name in _fieldnames():
            if name in first:
                macro[name] = first[name]
        macro["category"] = "macro_mean"
        for field in ["image_auc", "image_ap", "pixel_auc", "aupro_30", "aupro_30_percent", "approx_infer_flops_per_image"]:
            macro[field] = _finite_mean(float(row[field]) for row in group)
        macro["n_train_images"] = int(sum(int(row["n_train_images"]) for row in group))
        macro["n_test_images"] = int(sum(int(row["n_test_images"]) for row in group))
        macro["n_test_anomaly"] = int(sum(int(row["n_test_anomaly"]) for row in group))
        out.append(macro)
    return out


def _sort_ranking(rows: list[dict], target_metric: str) -> list[dict]:
    def key(row: dict) -> float:
        value = float(row.get(target_metric, float("nan")))
        return value if np.isfinite(value) else -float("inf")

    return sorted(rows, key=key, reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Architecture sweep for MVTec Feature-PILAR-FCN.")
    parser.add_argument("--data_root", default="data/mvtec_ad")
    parser.add_argument("--categories", default="screw,carpet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps_if_available")
    parser.add_argument("--output_dir", default="runs/benchmark/mvtec_feature_lipnet_fcn_arch_sweep_v1")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target_metric", default="image_auc", choices=["image_auc", "image_ap", "pixel_auc", "aupro_30"])
    parser.add_argument("--architecture_preset", default="quick", choices=["none", "quick", "full"])
    parser.add_argument("--architectures", nargs="*", default=[])
    parser.add_argument("--image_resize", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--backbone_weights", default="default", choices=["default", "none"])
    parser.add_argument("--feature_layers", default="layer2,layer3")
    parser.add_argument("--score_norms", nargs="+", default=["zscore"], choices=["none", "zscore", "quantile"])
    parser.add_argument("--hard_loss_settings", nargs="+", default=["0.10:1.0"])
    parser.add_argument("--image_topk_fracs", nargs="+", type=float, default=[0.005, 0.01, 0.02])
    parser.add_argument("--feature_normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature_encoder_channels", default="32,32")
    parser.add_argument("--feature_rf_schedule", default="3,3")
    parser.add_argument("--feature_overlap_schedule", default="2,2")
    parser.add_argument("--feature_decoder_channels", default="64,32")
    parser.add_argument("--inhibition_h", type=int, default=1)
    parser.add_argument("--refine_decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--l1_weight", type=float, default=0.0)
    parser.add_argument("--quantile_low", type=float, default=0.90)
    parser.add_argument("--quantile_high", type=float, default=0.995)
    parser.add_argument("--image_agg", default="topk_mean", choices=["topk_mean", "max", "mean"])
    parser.add_argument("--pro_eval_size", type=int, default=256)
    parser.add_argument("--fpr_limit", type=float, default=0.30)
    parser.add_argument("--num_thresholds", type=int, default=300)
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_test_images", type=int, default=0)
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    mvtec_root = _find_mvtec_root(Path(args.data_root))
    categories = _category_names(mvtec_root, args.categories)
    hard_settings = [_parse_hard_setting(x) for x in args.hard_loss_settings]
    architectures = _preset_architectures(args.architecture_preset)
    architectures.extend(_parse_architecture(x) for x in args.architectures)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"MVTec Feature-PILAR-FCN architecture sweep root={mvtec_root} categories={categories} "
        f"device={device} architectures={[a.name for a in architectures]} hard={args.hard_loss_settings} "
        f"score_norms={args.score_norms} topk={args.image_topk_fracs}",
        flush=True,
    )

    rows: list[dict] = []
    for arch in architectures:
        for hard_setting in hard_settings:
            for category in categories:
                rows.extend(_run_category_architecture(category, mvtec_root, arch, hard_setting, args, device))

    macro = _macro_rows(rows)
    ranking = _sort_ranking(macro, str(args.target_metric))
    _write_csv(out_dir / "summary_arch_sweep.csv", rows + macro, _fieldnames())
    _write_csv(out_dir / "ranking_arch_sweep.csv", ranking, _fieldnames())

    print(f"Saved: {out_dir / 'summary_arch_sweep.csv'}", flush=True)
    print(f"Saved: {out_dir / 'ranking_arch_sweep.csv'}", flush=True)
    if ranking:
        best = ranking[0]
        print(
            f"Best macro {args.target_metric}: arch={best['architecture']} hard={best['hard_loss']} "
            f"norm={best['score_norm']} topk={float(best['image_topk_frac']):g} "
            f"image_auc={float(best['image_auc']):.4f} pixel_auc={float(best['pixel_auc']):.4f} "
            f"aupro@{float(best['fpr_limit']):.2f}={float(best['aupro_30']):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
