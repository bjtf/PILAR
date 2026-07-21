#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from run_mvtec_feature_lipnet_fcn_sweep import (
    MVTecImageDataset,
    ResNetMultiFeatureExtractor,
    _aggregate_map_score,
    _build_models,
    _collect_train_maps,
    _make_stats,
    _parse_hard_setting,
    _parse_layer_set,
    _score_image,
    _train_models,
)
from run_mvtec_patch_lipnet import _category_names, _find_mvtec_root, _list_images, _load_mask, _load_rgb, _test_images
from run_mvtec_patch_lipnet import _image_to_tensor as _rgb_to_tensor
from lipnet.utils import resolve_device, set_seed


def _norm01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(np.nanpercentile(arr, 1.0))
    hi = float(np.nanpercentile(arr, 99.0))
    if hi <= lo + 1e-8:
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
    if hi <= lo + 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _heat_to_rgb(heat: np.ndarray) -> Image.Image:
    h = _norm01(heat)
    r = np.clip(2.0 * h, 0.0, 1.0)
    g = np.clip(2.0 * (1.0 - np.abs(h - 0.5) * 2.0), 0.0, 1.0)
    b = np.clip(2.0 * (1.0 - h), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((255.0 * rgb).astype(np.uint8), mode="RGB")


def _overlay(img: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = img.convert("RGB")
    heat_img = _heat_to_rgb(heat).resize(base.size, Image.BILINEAR)
    return Image.blend(base, heat_img, float(alpha))


def _mask_to_rgb(mask: np.ndarray, size: tuple[int, int]) -> Image.Image:
    if mask is None:
        return Image.new("RGB", size, (0, 0, 0))
    arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L").resize(size, Image.NEAREST).convert("RGB")


def _label_bar(text: str, width: int, height: int = 34) -> Image.Image:
    out = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((8, 10), text, fill=(20, 20, 20), font=font)
    return out


def _panel(
    img: Image.Image,
    heat: np.ndarray,
    mask: np.ndarray | None,
    title: str,
    output: Path,
) -> None:
    img = img.convert("RGB")
    heat_img = _heat_to_rgb(heat).resize(img.size, Image.BILINEAR)
    overlay = _overlay(img, heat)
    mask_img = _mask_to_rgb(mask, img.size)
    tiles = [
        ("input", img),
        ("anomaly map", heat_img),
        ("overlay", overlay),
        ("ground truth", mask_img),
    ]
    label_h = 34
    title_h = 42
    w, h = img.size
    out = Image.new("RGB", (w * len(tiles), h + label_h + title_h), (255, 255, 255))
    out.paste(_label_bar(title, out.size[0], title_h), (0, 0))
    for i, (label, tile) in enumerate(tiles):
        x = i * w
        out.paste(_label_bar(label, w, label_h), (x, title_h))
        out.paste(tile, (x, title_h + label_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    out.save(output)


def _select_examples(results: list[dict], category: str, max_regular: int) -> list[dict]:
    anomalies = [r for r in results if int(r["label"]) == 1]
    normals = [r for r in results if int(r["label"]) == 0]
    selected: list[dict] = []
    if anomalies:
        selected.append(max(anomalies, key=lambda r: float(r["score"])) | {"selection": "top_anomaly"})
    if category == "screw":
        if anomalies:
            selected.append(min(anomalies, key=lambda r: float(r["score"])) | {"selection": "lowest_anomaly"})
        if normals:
            selected.append(max(normals, key=lambda r: float(r["score"])) | {"selection": "highest_normal"})
    return selected[:max_regular]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "category",
        "selection",
        "image",
        "defect_type",
        "label",
        "score",
        "mask_found",
        "panel",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qualitative MVTec Feature-LIPNet-FCN panels.")
    parser.add_argument("--data_root", default="data/mvtec_ad")
    parser.add_argument("--categories", default="bottle,leather,hazelnut,toothbrush,zipper,screw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps_if_available")
    parser.add_argument("--output_dir", default="runs/benchmark/mvtec_qualitative_step1")
    parser.add_argument("--image_resize", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--backbone_weights", default="default", choices=["default", "none"])
    parser.add_argument("--feature_layers", default="layer2,layer3")
    parser.add_argument("--score_norm", default="zscore", choices=["none", "zscore", "quantile"])
    parser.add_argument("--hard_loss", default="0.10:1.0")
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
    parser.add_argument("--image_topk_frac", type=float, default=0.01)
    parser.add_argument("--max_examples_per_category", type=int, default=3)
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    mvtec_root = _find_mvtec_root(Path(args.data_root))
    categories = _category_names(mvtec_root, args.categories)
    layers = _parse_layer_set(args.feature_layers)
    hard_frac, hard_weight = _parse_hard_setting(args.hard_loss)
    out_dir = Path(args.output_dir)
    summary_rows: list[dict] = []

    print(f"Generating MVTec qualitative panels root={mvtec_root} categories={categories} device={device}", flush=True)
    for category in categories:
        category_root = mvtec_root / category
        train_paths = _list_images(category_root / "train" / "good")
        test_rows = _test_images(category_root)
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
        print(f"{category}: train={len(train_paths)} test={len(test_rows)} shapes={shapes}", flush=True)
        _train_models(models, backbone, train_loader, device, args, hard_frac, hard_weight)
        train_maps = _collect_train_maps(models, backbone, train_loader, device, args)
        stats = _make_stats(train_maps, str(args.score_norm), float(args.quantile_low), float(args.quantile_high))

        scored: list[dict] = []
        for item in test_rows:
            img = _load_rgb(item.path, image_resize=int(args.image_resize))
            heat = _score_image(models, backbone, _rgb_to_tensor(img), stats, device, args)
            score = _aggregate_map_score(heat, str(args.image_agg), float(args.image_topk_frac))
            scored.append(
                {
                    "category": category,
                    "image": str(item.path.relative_to(category_root)),
                    "path": item.path,
                    "defect_type": item.defect_type,
                    "label": int(item.label),
                    "score": float(score),
                    "mask_path": item.mask_path,
                    "mask_found": int(item.mask_path is not None),
                    "heat": heat,
                }
            )

        for row in _select_examples(scored, category, int(args.max_examples_per_category)):
            img = _load_rgb(row["path"], image_resize=int(args.image_resize))
            mask = _load_mask(row["mask_path"], (int(args.image_resize), int(args.image_resize)))
            safe_image = Path(str(row["image"]).replace("/", "__")).stem
            panel_path = out_dir / category / f"{row['selection']}_{row['defect_type']}_{safe_image}.png"
            title = f"{category} | {row['selection']} | {row['defect_type']} | score={float(row['score']):.4f}"
            _panel(img, row["heat"], mask, title, panel_path)
            out_row = dict(row)
            out_row["panel"] = str(panel_path)
            out_row.pop("heat", None)
            out_row.pop("path", None)
            out_row.pop("mask_path", None)
            summary_rows.append(out_row)
            print(f" saved {panel_path}", flush=True)

    _write_csv(out_dir / "qualitative_examples.csv", summary_rows)
    print(f"Saved summary: {out_dir / 'qualitative_examples.csv'}", flush=True)


if __name__ == "__main__":
    main()
