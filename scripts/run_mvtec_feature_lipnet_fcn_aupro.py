#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import math
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from lipnet.efficiency import estimate_forward_flops_per_sample, model_size_metrics
from lipnet.metrics import pr_auc, roc_auc
from lipnet.utils import resolve_device, set_seed
from scripts.run_mvtec_feature_lipnet_fcn_sweep import (
    MVTecImageDataset,
    ResNetMultiFeatureExtractor,
    _aggregate_map_score,
    _build_models,
    _collect_train_maps,
    _make_stats,
    _parse_hard_setting,
    _parse_layer_set,
    _safe_label,
    _score_image,
    _train_models,
)
from scripts.run_mvtec_patch_lipnet import (
    _category_names,
    _find_mvtec_root,
    _list_images,
    _load_mask,
    _load_rgb,
    _resize_array,
    _test_images,
    _write_csv,
)
from scripts.run_mvtec_patch_lipnet import _image_to_tensor as _rgb_to_tensor


from lipnet.pro_metrics import (
    connected_regions as _connected_regions,
    thresholds_from_maps as _thresholds_from_maps,
    clip_curve_at_fpr as _clip_curve_at_fpr,
    aupro_at_fpr as _aupro_at_fpr,
)


def _count_params(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = [p for p in params if p.requires_grad]
    return int(sum(p.numel() for p in params))


def _evaluate_norm(
    category: str,
    category_root: Path,
    test_rows: list,
    models: dict[str, nn.Module],
    backbone: nn.Module,
    train_maps: dict[str, np.ndarray],
    layers_label: str,
    hard_label: str,
    score_norm: str,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    stats = _make_stats(train_maps, score_norm, float(args.quantile_low), float(args.quantile_high))

    image_results = []
    pixel_scores: List[np.ndarray] = []
    pixel_labels: List[np.ndarray] = []
    heat_maps: list[np.ndarray] = []
    mask_maps: list[np.ndarray] = []

    for i, item in enumerate(test_rows, start=1):
        img = _load_rgb(item.path, image_resize=int(args.image_resize))
        x = _rgb_to_tensor(img)
        heat = _score_image(models, backbone, x, stats, device, args)
        image_score = _aggregate_map_score(heat, str(args.image_agg), float(args.image_topk_frac))
        image_results.append(
            {
                "category": category,
                "image": str(item.path.relative_to(category_root)),
                "defect_type": item.defect_type,
                "label": int(item.label),
                "score": float(image_score),
                "mask_found": int(item.mask_path is not None),
                "feature_layers": layers_label,
                "hard_loss": hard_label,
                "score_norm": score_norm,
            }
        )

        mask = _load_mask(item.mask_path, (int(args.image_resize), int(args.image_resize)))
        eval_size = int(args.pro_eval_size)
        if eval_size > 0:
            heat_eval = _resize_array(heat.astype(np.float32), eval_size, Image.BILINEAR).astype(np.float32)
            mask_eval = _resize_array((mask * 255).astype(np.uint8), eval_size, Image.NEAREST) > 0
        else:
            heat_eval = heat.astype(np.float32)
            mask_eval = mask > 0

        heat_maps.append(heat_eval)
        mask_maps.append(mask_eval)
        pixel_scores.append(heat_eval.reshape(-1))
        pixel_labels.append(mask_eval.astype(np.uint8).reshape(-1))

        if i % max(1, int(args.progress_every)) == 0:
            print(
                f" {category}/{layers_label}/{hard_label}/{score_norm}: evaluated {i}/{len(test_rows)}",
                flush=True,
            )

    y = np.asarray([r["label"] for r in image_results], dtype=np.int64)
    s = np.asarray([r["score"] for r in image_results], dtype=np.float64)
    image_auc = roc_auc(s, y)
    image_ap = pr_auc(s, y)
    pixel_auc = roc_auc(np.concatenate(pixel_scores), np.concatenate(pixel_labels))
    pro_info, raw_curve, clipped_curve = _aupro_at_fpr(
        heat_maps,
        mask_maps,
        fpr_limit=float(args.fpr_limit),
        num_thresholds=int(args.num_thresholds),
        connectivity=int(args.connectivity),
    )

    combo_label = f"{_safe_label(layers_label)}_hard{_safe_label(hard_label)}_{score_norm}"
    if getattr(args, "save_artifacts", False):
        from lipnet.checkpoint import export_inference
        directory = Path(args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        export_inference(models, backbone, stats, args, directory / f"model_{category}_{combo_label}.pt")
        path = directory / f"maps_{category}_{combo_label}.npz"
        with path.with_suffix(".tmp").open("wb") as f:
            np.savez_compressed(f, heat_maps=np.stack(heat_maps), masks=np.stack(mask_maps),
                                images=np.asarray([x["image"] for x in image_results]),
                                labels=y, scores=s)
        path.with_suffix(".tmp").replace(path)
    _write_csv(
        Path(args.output_dir) / f"results_{category}_{combo_label}.csv",
        image_results,
        ["category", "image", "defect_type", "label", "score", "mask_found", "feature_layers", "hard_loss", "score_norm"],
    )
    _write_csv(
        Path(args.output_dir) / f"pro_curve_raw_{category}_{combo_label}.csv",
        raw_curve,
        ["threshold", "fpr", "pro"],
    )
    _write_csv(
        Path(args.output_dir) / f"pro_curve_clipped_{category}_{combo_label}.csv",
        clipped_curve,
        ["fpr", "pro"],
    )

    row = {
        "category": category,
        "feature_layers": layers_label,
        "hard_loss": hard_label,
        "score_norm": score_norm,
        "image_auc": float(image_auc),
        "image_ap": float(image_ap),
        "pixel_auc": float(pixel_auc),
        "aupro_30": float(pro_info["aupro"]),
        "aupro_30_percent": float(pro_info["aupro"]) * 100.0,
        "fpr_limit": float(pro_info["fpr_limit"]),
        "num_thresholds": int(pro_info["num_thresholds"]),
        "n_pro_regions": int(pro_info["n_regions"]),
        "n_normal_pixels": int(pro_info["n_normal_pixels"]),
        "n_test_images": int(len(test_rows)),
        "n_test_anomaly": int(y.sum()),
        "pro_eval_size": int(args.pro_eval_size),
        "connectivity": int(args.connectivity),
    }
    print(
        f"{category}/{layers_label}/hard={hard_label}/norm={score_norm}: "
        f"image_auc={image_auc:.4f} pixel_auc={pixel_auc:.4f} aupro@{float(args.fpr_limit):.2f}={float(pro_info['aupro']):.4f}",
        flush=True,
    )
    return row


def _run_category(
    category: str,
    mvtec_root: Path,
    layers: List[str],
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

    train_loader = DataLoader(
        MVTecImageDataset(train_paths, image_resize=int(args.image_resize)),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=False,
    )
    sample = next(iter(train_loader))[:1]
    backbone = ResNetMultiFeatureExtractor(args.backbone, layers, args.backbone_weights).to(device).eval()
    models, shapes = _build_models(backbone, layers, sample, device, args)

    layers_label = ",".join(layers)
    hard_frac, hard_weight = hard_setting
    hard_label = f"{hard_frac:g}:{hard_weight:g}"
    print(
        f"{category}: layers={layers_label} hard={hard_label} train_images={len(train_paths)} "
        f"test_images={len(test_rows)} shapes={shapes}",
        flush=True,
    )

    _train_models(models, backbone, train_loader, device, args, hard_frac, hard_weight)
    train_maps = _collect_train_maps(models, backbone, train_loader, device, args)

    backbone_flops = estimate_forward_flops_per_sample(backbone, (3, int(args.image_resize), int(args.image_resize)))
    backbone_params = _count_params(backbone, trainable_only=False)
    lipnet_params = int(sum(model_size_metrics(model)["param_count"] for model in models.values()))
    lipnet_bytes = int(sum(model_size_metrics(model)["param_bytes"] for model in models.values()))
    lipnet_flops = float(
        sum(estimate_forward_flops_per_sample(model, shapes[layer])["flops_forward_total"] for layer, model in models.items())
    )

    rows = []
    for score_norm in args.score_norms:
        row = _evaluate_norm(
            category,
            category_root,
            test_rows,
            models,
            backbone,
            train_maps,
            layers_label,
            hard_label,
            str(score_norm),
            device,
            args,
        )
        row.update(
            {
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
        rows.append(row)
    return rows


def _summary_fieldnames() -> list[str]:
    return [
        "category",
        "feature_layers",
        "hard_loss",
        "score_norm",
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
        "train_topk_feature_frac",
        "train_topk_feature_weight",
        "quantile_low",
        "quantile_high",
    ]


def _finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray([float(x) for x in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _macro_row(rows: list[dict]) -> dict:
    numeric_mean_fields = [
        "image_auc",
        "image_ap",
        "pixel_auc",
        "aupro_30",
        "aupro_30_percent",
        "approx_infer_flops_per_image",
    ]
    out = {name: "" for name in _summary_fieldnames()}
    out["category"] = "macro_mean"
    if rows:
        for field in numeric_mean_fields:
            out[field] = _finite_mean(float(r[field]) for r in rows)
        for field in ["feature_layers", "hard_loss", "score_norm", "fpr_limit", "num_thresholds", "image_resize", "pro_eval_size", "connectivity", "backbone"]:
            out[field] = rows[0].get(field, "")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MVTec Feature-PILAR-FCN AUPRO at bounded FPR.")
    parser.add_argument("--data_root", default="data/mvtec_ad")
    parser.add_argument("--categories", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps_if_available")
    parser.add_argument("--output_dir", default="runs/benchmark/mvtec_feature_lipnet_fcn_layer23_hard_aupro30_v1")
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
    parser.add_argument("--hard_loss", default="0.10:1.0")
    parser.add_argument("--feature_normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature_encoder_channels", default="32,32")
    parser.add_argument("--feature_rf_schedule", default="3,3")
    parser.add_argument("--feature_overlap_schedule", default="2,2")
    parser.add_argument("--feature_decoder_channels", default="64,32")
    parser.add_argument("--feature_inhibition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inhibition_h", type=int, default=1)
    parser.add_argument("--refine_decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--l1_weight", type=float, default=0.0)
    parser.add_argument("--quantile_low", type=float, default=0.90)
    parser.add_argument("--quantile_high", type=float, default=0.995)
    parser.add_argument("--image_agg", default="topk_mean", choices=["topk_mean", "max", "mean"])
    parser.add_argument("--image_topk_frac", type=float, default=0.01)
    parser.add_argument("--pro_eval_size", type=int, default=256)
    parser.add_argument("--fpr_limit", type=float, default=0.30)
    parser.add_argument("--num_thresholds", type=int, default=400)
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_test_images", type=int, default=0)
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    mvtec_root = _find_mvtec_root(Path(args.data_root))
    categories = _category_names(mvtec_root, args.categories)
    layers = _parse_layer_set(args.feature_layers)
    hard_setting = _parse_hard_setting(args.hard_loss)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"MVTec Feature-PILAR-FCN AUPRO root={mvtec_root} categories={categories} device={device} "
        f"layers={layers} hard={args.hard_loss} norms={args.score_norms} fpr_limit={args.fpr_limit}",
        flush=True,
    )

    rows: list[dict] = []
    for category in categories:
        rows.extend(_run_category(category, mvtec_root, layers, hard_setting, args, device))

    summary_rows = rows + [_macro_row(rows)]
    _write_csv(out_dir / "summary_aupro.csv", summary_rows, _summary_fieldnames())

    print(f"Saved: {out_dir / 'summary_aupro.csv'}", flush=True)
    if rows:
        print(
            f"Macro: image_auc={_finite_mean(float(r['image_auc']) for r in rows):.4f} "
            f"pixel_auc={_finite_mean(float(r['pixel_auc']) for r in rows):.4f} "
            f"aupro@{float(args.fpr_limit):.2f}={_finite_mean(float(r['aupro_30']) for r in rows):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
