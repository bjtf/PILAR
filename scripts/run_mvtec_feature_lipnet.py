#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import csv
import math
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from typing import Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

from lipnet.efficiency import estimate_forward_flops_per_sample, model_size_metrics
from lipnet.metrics import pr_auc, roc_auc
from lipnet.one_class import LIPNetAutoencoder
from lipnet.utils import resolve_device, set_seed
from scripts.run_mvtec_patch_lipnet import (
    MVTecPatchDataset,
    _aggregate_patch_scores,
    _category_names,
    _find_mvtec_root,
    _heatmap_from_patches,
    _image_patches,
    _list_images,
    _load_mask,
    _make_train_refs,
    _resize_array,
    _test_images,
    _write_csv,
)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone: str, layer: str, weights: str) -> None:
        super().__init__()
        backbone = str(backbone).lower()
        layer = str(layer).lower()
        weights = str(weights).lower()

        if backbone == "resnet18":
            w = ResNet18_Weights.DEFAULT if weights == "default" else None
            model = resnet18(weights=w)
        elif backbone == "resnet50":
            w = ResNet50_Weights.DEFAULT if weights == "default" else None
            model = resnet50(weights=w)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        blocks: List[nn.Module] = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1]
        if layer in {"layer2", "layer3", "layer4"}:
            blocks.append(model.layer2)
        if layer in {"layer3", "layer4"}:
            blocks.append(model.layer3)
        if layer == "layer4":
            blocks.append(model.layer4)
        if layer not in {"layer1", "layer2", "layer3", "layer4"}:
            raise ValueError("--feature_layer must be one of layer1, layer2, layer3, layer4")

        self.net = nn.Sequential(*blocks)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.net(x)


def _parse_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _make_feature_model(args: argparse.Namespace, feature_shape: tuple[int, int, int]) -> LIPNetAutoencoder:
    c, h, w = map(int, feature_shape)
    if h != w:
        raise ValueError(f"Feature map must be square for LIPNetAutoencoder, got {h}x{w}")

    channels = _parse_ints(args.feature_encoder_channels)
    rfs = _parse_ints(args.feature_rf_schedule)
    overlaps = _parse_ints(args.feature_overlap_schedule)
    if not (len(channels) == len(rfs) == len(overlaps)):
        raise ValueError("--feature_encoder_channels, --feature_rf_schedule, and --feature_overlap_schedule must match length")

    inhibition_enabled = bool(getattr(args, "feature_inhibition", True))
    layers = []
    for i, (ch, rf, ov) in enumerate(zip(channels, rfs, overlaps)):
        layers.append(
            {
                "rf": int(rf),
                "overlap": int(ov),
                "activation": "relu",
                "channels": int(ch),
                "inhibition": {
                    "enabled": inhibition_enabled,
                    "mode": "fixed" if i == 0 else "radial",
                    "delta": 1.0,
                    "h": int(args.inhibition_h),
                    "radial_init_mu": 1.0,
                    "radial_init_sigma": 1.0,
                },
            }
        )

    dec = {
        "type": "pure_pyramid",
        "channels": _parse_ints(args.feature_decoder_channels),
        "activation": "relu",
        "use_upsample": True,
        "upsample_mode": "bilinear",
    }
    if args.refine_decoder:
        dec["refine"] = {"type": "depthwise", "activation": "relu", "residual": True}

    return LIPNetAutoencoder(
        input_size=h,
        input_channels=c,
        output_channels=c,
        layers_2d=layers,
        decoder_hidden=[],
        decoder_cfg=dec,
    )


def _feature_map(backbone: nn.Module, x: torch.Tensor, normalize: bool) -> torch.Tensor:
    with torch.no_grad():
        feat = backbone(x)
    if normalize:
        feat = F.normalize(feat, p=2, dim=1)
    return feat


def _feature_loss_and_scores(
    model: nn.Module,
    feat: torch.Tensor,
    l1_weight: float,
    topk_frac: float,
    topk_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    recon = model(feat)
    residual = recon - feat
    mse = residual.square().mean(dim=1)
    l1 = residual.abs().mean(dim=1)
    loc = mse + float(l1_weight) * l1
    mean_score = loc.mean(dim=(1, 2))
    if topk_frac > 0.0 and topk_weight != 0.0:
        flat = loc.flatten(1)
        k = max(1, int(math.ceil(float(topk_frac) * flat.shape[1])))
        topk_score = flat.topk(k, dim=1).values.mean(dim=1)
        score = mean_score + float(topk_weight) * topk_score
    else:
        score = mean_score
    return mean_score.mean(), score


def _train(
    model: nn.Module,
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    for epoch in range(1, int(args.epochs) + 1):
        total = 0.0
        n = 0
        for x in loader:
            x = x.to(device)
            feat = _feature_map(backbone, x, bool(args.feature_normalize))
            opt.zero_grad(set_to_none=True)
            loss, _ = _feature_loss_and_scores(
                model,
                feat,
                l1_weight=float(args.l1_weight),
                topk_frac=float(args.train_topk_feature_frac),
                topk_weight=float(args.train_topk_feature_weight),
            )
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(x.shape[0])
            n += int(x.shape[0])
        print(f" epoch={epoch}/{args.epochs} loss={total / max(1, n):.6f}", flush=True)


def _score_loader(model: nn.Module, backbone: nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            feat = _feature_map(backbone, x, bool(args.feature_normalize))
            _, s = _feature_loss_and_scores(
                model,
                feat,
                l1_weight=float(args.l1_weight),
                topk_frac=float(args.score_topk_feature_frac),
                topk_weight=float(args.score_topk_feature_weight),
            )
            out.append(s.cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


def _score_patches(model: nn.Module, backbone: nn.Module, patches: torch.Tensor, device: torch.device, args: argparse.Namespace) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    batch_size = int(args.batch_size)
    with torch.no_grad():
        for start in range(0, int(patches.shape[0]), batch_size):
            x = patches[start : start + batch_size].to(device)
            feat = _feature_map(backbone, x, bool(args.feature_normalize))
            _, s = _feature_loss_and_scores(
                model,
                feat,
                l1_weight=float(args.l1_weight),
                topk_frac=float(args.score_topk_feature_frac),
                topk_weight=float(args.score_topk_feature_weight),
            )
            out.append(s.cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


def _backbone_param_count(backbone: nn.Module) -> int:
    return int(sum(p.numel() for p in backbone.parameters()))


def _run_category(category: str, mvtec_root: Path, args: argparse.Namespace, device: torch.device) -> dict:
    category_root = mvtec_root / category
    train_paths = _list_images(category_root / "train" / "good")
    test_rows = _test_images(category_root)
    refs = _make_train_refs(
        train_paths,
        patch_size=int(args.patch_size),
        stride=int(args.train_patch_stride),
        max_patches=int(args.max_train_patches),
        seed=int(args.seed),
        image_resize=int(args.image_resize),
    )
    print(f"{category}: train_images={len(train_paths)} train_patches={len(refs)} test_images={len(test_rows)}")

    loader = DataLoader(
        MVTecPatchDataset(refs, patch_size=int(args.patch_size), image_resize=int(args.image_resize)),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=False,
    )

    backbone = ResNetFeatureExtractor(args.backbone, args.feature_layer, args.backbone_weights).to(device).eval()
    sample = next(iter(loader))[:1].to(device)
    sample_feat = _feature_map(backbone, sample, bool(args.feature_normalize))
    feature_shape = tuple(map(int, sample_feat.shape[1:]))

    model = _make_feature_model(args, feature_shape).to(device)
    size = model_size_metrics(model)
    feature_flops = estimate_forward_flops_per_sample(model, feature_shape)
    backbone_flops = estimate_forward_flops_per_sample(backbone, (3, int(args.patch_size), int(args.patch_size)))

    _train(model, backbone, loader, device, args)
    train_scores = _score_loader(model, backbone, loader, device, args)
    train_mu = float(train_scores.mean()) if train_scores.size else 0.0
    train_sd = float(train_scores.std()) if train_scores.size else 1.0
    if train_sd < 1e-8:
        train_sd = 1.0

    image_results = []
    pixel_scores: List[np.ndarray] = []
    pixel_labels: List[np.ndarray] = []
    for i, item in enumerate(test_rows, start=1):
        patches, coords, image_size = _image_patches(
            item.path,
            int(args.patch_size),
            int(args.test_patch_stride),
            int(args.image_resize),
        )
        patch_scores = _score_patches(model, backbone, patches, device, args)
        if args.score_znorm:
            patch_scores = (patch_scores - train_mu) / train_sd
        image_score = _aggregate_patch_scores(patch_scores, str(args.image_agg), float(args.image_topk_frac))
        image_results.append(
            {
                "category": category,
                "image": str(item.path.relative_to(category_root)),
                "defect_type": item.defect_type,
                "label": int(item.label),
                "score": float(image_score),
                "n_patches": int(patch_scores.size),
                "mask_found": int(item.mask_path is not None),
            }
        )
        if args.eval_pixel_auc:
            heat = _heatmap_from_patches(patch_scores, coords, image_size, int(args.patch_size))
            mask = _load_mask(item.mask_path, image_size)
            if int(args.pixel_auc_size) > 0:
                heat = _resize_array(heat.astype(np.float32), int(args.pixel_auc_size), 2)
                mask = _resize_array((mask * 255).astype(np.uint8), int(args.pixel_auc_size), 0) > 0
            pixel_scores.append(heat.astype(np.float32).reshape(-1))
            pixel_labels.append(mask.astype(np.uint8).reshape(-1))
        if i % max(1, int(args.progress_every)) == 0:
            print(f" {category}: evaluated {i}/{len(test_rows)}", flush=True)

    y = np.array([r["label"] for r in image_results], dtype=np.int64)
    s = np.array([r["score"] for r in image_results], dtype=np.float64)
    image_auc = roc_auc(s, y)
    image_ap = pr_auc(s, y)
    pixel_auc = float("nan")
    if args.eval_pixel_auc and pixel_scores:
        pixel_auc = roc_auc(np.concatenate(pixel_scores), np.concatenate(pixel_labels))
    avg_test_patches = float(np.mean([float(r["n_patches"]) for r in image_results])) if image_results else float("nan")

    out_dir = Path(args.output_dir)
    _write_csv(
        out_dir / f"results_{category}.csv",
        image_results,
        ["category", "image", "defect_type", "label", "score", "n_patches", "mask_found"],
    )

    row = {
        "category": category,
        "image_auc": float(image_auc),
        "image_ap": float(image_ap),
        "pixel_auc": float(pixel_auc),
        "n_train_images": int(len(train_paths)),
        "n_train_patches": int(len(refs)),
        "n_test_images": int(len(test_rows)),
        "n_test_anomaly": int(y.sum()),
        "feature_shape": "x".join(map(str, feature_shape)),
        "lipnet_params": int(size["param_count"]),
        "lipnet_param_bytes": int(size["param_bytes"]),
        "backbone_params_frozen": _backbone_param_count(backbone),
        "backbone_flops_per_patch": float(backbone_flops["flops_forward_total"]),
        "lipnet_flops_per_patch": float(feature_flops["flops_forward_total"]),
        "avg_test_patches": float(avg_test_patches),
        "approx_infer_flops_per_image": float(
            avg_test_patches * (float(backbone_flops["flops_forward_total"]) + float(feature_flops["flops_forward_total"]))
        ),
        "train_score_mu": float(train_mu),
        "train_score_sd": float(train_sd),
    }
    print(
        f"{category}: image_auc={image_auc:.4f} image_ap={image_ap:.4f} "
        f"pixel_auc={pixel_auc:.4f} lipnet_params={row['lipnet_params']} feature_shape={row['feature_shape']}",
        flush=True,
    )
    return row


def _write_summary(path: Path, rows: Iterable[dict]) -> None:
    fieldnames = [
        "category",
        "image_auc",
        "image_ap",
        "pixel_auc",
        "n_train_images",
        "n_train_patches",
        "n_test_images",
        "n_test_anomaly",
        "feature_shape",
        "lipnet_params",
        "lipnet_param_bytes",
        "backbone_params_frozen",
        "backbone_flops_per_patch",
        "lipnet_flops_per_patch",
        "avg_test_patches",
        "approx_infer_flops_per_image",
        "train_score_mu",
        "train_score_sd",
    ]
    _write_csv(path, rows, fieldnames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-space Patch-LIPNet benchmark for MVTec AD.")
    parser.add_argument("--data_root", default="data/mvtec_ad")
    parser.add_argument("--categories", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps_if_available")
    parser.add_argument("--output_dir", default="runs/benchmark/mvtec_feature_lipnet_v1")
    parser.add_argument("--image_resize", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--train_patch_stride", type=int, default=64)
    parser.add_argument("--test_patch_stride", type=int, default=64)
    parser.add_argument("--max_train_patches", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--backbone_weights", default="default", choices=["default", "none"])
    parser.add_argument("--feature_layer", default="layer2", choices=["layer1", "layer2", "layer3", "layer4"])
    parser.add_argument("--feature_normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature_encoder_channels", default="32,32")
    parser.add_argument("--feature_rf_schedule", default="3,3")
    parser.add_argument("--feature_overlap_schedule", default="2,2")
    parser.add_argument("--feature_decoder_channels", default="64,32")
    parser.add_argument("--feature_inhibition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inhibition_h", type=int, default=1)
    parser.add_argument("--refine_decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--l1_weight", type=float, default=0.0)
    parser.add_argument("--train_topk_feature_frac", type=float, default=0.0)
    parser.add_argument("--train_topk_feature_weight", type=float, default=0.0)
    parser.add_argument("--score_topk_feature_frac", type=float, default=0.10)
    parser.add_argument("--score_topk_feature_weight", type=float, default=1.0)
    parser.add_argument("--score_znorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image_agg", default="topk_mean", choices=["topk_mean", "max", "mean"])
    parser.add_argument("--image_topk_frac", type=float, default=0.10)
    parser.add_argument("--eval_pixel_auc", action="store_true")
    parser.add_argument("--pixel_auc_size", type=int, default=256)
    parser.add_argument("--progress_every", type=int, default=25)
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    mvtec_root = _find_mvtec_root(Path(args.data_root))
    categories = _category_names(mvtec_root, args.categories)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"MVTec Feature-LIPNet root={mvtec_root} categories={categories} "
        f"device={device} backbone={args.backbone}:{args.feature_layer}"
    )

    rows = []
    for category in categories:
        rows.append(_run_category(category, mvtec_root, args, device))
    _write_summary(out_dir / "summary.csv", rows)

    image_aucs = np.array([r["image_auc"] for r in rows], dtype=np.float64)
    pixel_aucs = np.array([r["pixel_auc"] for r in rows], dtype=np.float64)
    print("Summary:")
    print(f" mean_image_auc={np.nanmean(image_aucs):.4f}")
    if np.isfinite(pixel_aucs).any():
        print(f" mean_pixel_auc={np.nanmean(pixel_aucs):.4f}")
    print(f" saved={out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
