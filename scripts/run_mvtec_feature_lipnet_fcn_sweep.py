#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

from lipnet.efficiency import estimate_forward_flops_per_sample, model_size_metrics
from lipnet.metrics import pr_auc, roc_auc
from lipnet.one_class import LIPNetAutoencoder
from lipnet.utils import resolve_device, set_seed
from run_mvtec_patch_lipnet import (
    _category_names,
    _find_mvtec_root,
    _list_images,
    _load_mask,
    _load_rgb,
    _resize_array,
    _test_images,
    _write_csv,
)
from run_mvtec_patch_lipnet import _image_to_tensor as _rgb_to_tensor
from run_mvtec_feature_lipnet import _make_feature_model


@dataclass(frozen=True)
class LayerStats:
    mode: str
    mean: float
    std: float
    q_low: float
    q_high: float


class MVTecImageDataset(Dataset):
    def __init__(self, paths: List[Path], image_resize: int) -> None:
        self.paths = list(paths)
        self.image_resize = int(image_resize)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = _load_rgb(self.paths[idx], image_resize=self.image_resize)
        return _rgb_to_tensor(img)


class ResNetMultiFeatureExtractor(nn.Module):
    def __init__(self, backbone: str, layers: Iterable[str], weights: str) -> None:
        super().__init__()
        self.layers = list(dict.fromkeys(str(x).lower() for x in layers))
        allowed = {"layer1", "layer2", "layer3", "layer4"}
        bad = [x for x in self.layers if x not in allowed]
        if bad:
            raise ValueError(f"Unsupported feature layers: {bad}")

        backbone = str(backbone).lower()
        weights = str(weights).lower()
        if backbone == "resnet18":
            w = ResNet18_Weights.DEFAULT if weights == "default" else None
            model = resnet18(weights=w)
        elif backbone == "resnet50":
            w = ResNet50_Weights.DEFAULT if weights == "default" else None
            model = resnet50(weights=w)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.max_layer_idx = max(int(x[-1]) for x in self.layers)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2 if self.max_layer_idx >= 2 else None
        self.layer3 = model.layer3 if self.max_layer_idx >= 3 else None
        self.layer4 = model.layer4 if self.max_layer_idx >= 4 else None

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = (x - self.mean) / self.std
        x = self.stem(x)
        out: dict[str, torch.Tensor] = {}
        x = self.layer1(x)
        if "layer1" in self.layers:
            out["layer1"] = x
        if self.max_layer_idx <= 1:
            return out
        if self.layer2 is None:
            return out
        x = self.layer2(x)
        if "layer2" in self.layers:
            out["layer2"] = x
        if self.max_layer_idx <= 2:
            return out
        if self.layer3 is None:
            return out
        x = self.layer3(x)
        if "layer3" in self.layers:
            out["layer3"] = x
        if self.max_layer_idx <= 3:
            return out
        if self.layer4 is None:
            return out
        x = self.layer4(x)
        if "layer4" in self.layers:
            out["layer4"] = x
        return out


def _parse_layer_set(value: str) -> List[str]:
    layers = [x.strip().lower() for x in str(value).split(",") if x.strip()]
    if not layers:
        raise ValueError("Empty feature layer set")
    return layers


def _parse_hard_setting(value: str) -> tuple[float, float]:
    parts = [x.strip() for x in str(value).split(":")]
    if len(parts) != 2:
        raise ValueError(f"Hard setting must be frac:weight, got {value!r}")
    return float(parts[0]), float(parts[1])


def _safe_label(text: str) -> str:
    return (
        str(text)
        .replace(",", "+")
        .replace(":", "x")
        .replace("/", "_")
        .replace(" ", "")
        .replace(".", "p")
    )


def _feature_loss_and_map(
    model: nn.Module,
    feat: torch.Tensor,
    l1_weight: float,
    train_topk_feature_frac: float,
    train_topk_feature_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    recon = model(feat)
    residual = recon - feat
    loc = residual.square().mean(dim=1) + float(l1_weight) * residual.abs().mean(dim=1)
    mean_loss = loc.mean()
    if train_topk_feature_frac > 0.0 and train_topk_feature_weight != 0.0:
        flat = loc.flatten(1)
        k = max(1, int(math.ceil(float(train_topk_feature_frac) * flat.shape[1])))
        topk_loss = flat.topk(k, dim=1).values.mean()
        loss = mean_loss + float(train_topk_feature_weight) * topk_loss
    else:
        loss = mean_loss
    return loss, loc


def _normalize_feature(feat: torch.Tensor, enabled: bool) -> torch.Tensor:
    return F.normalize(feat, p=2, dim=1) if enabled else feat


def _train_models(
    models: dict[str, nn.Module],
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    train_topk_feature_frac: float,
    train_topk_feature_weight: float,
) -> None:
    for model in models.values():
        model.train()
    opt = torch.optim.Adam(
        [p for model in models.values() for p in model.parameters()],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    for epoch in range(1, int(args.epochs) + 1):
        total = 0.0
        n = 0
        for x in loader:
            x = x.to(device)
            with torch.no_grad():
                feats = backbone(x)
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), dtype=x.dtype, device=device)
            for layer, model in models.items():
                feat = _normalize_feature(feats[layer], bool(args.feature_normalize))
                layer_loss, _ = _feature_loss_and_map(
                    model,
                    feat,
                    l1_weight=float(args.l1_weight),
                    train_topk_feature_frac=float(train_topk_feature_frac),
                    train_topk_feature_weight=float(train_topk_feature_weight),
                )
                loss = loss + layer_loss
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(x.shape[0])
            n += int(x.shape[0])
        print(f" epoch={epoch}/{args.epochs} loss={total / max(1, n):.6f}", flush=True)


def _collect_train_maps(
    models: dict[str, nn.Module],
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    for model in models.values():
        model.eval()
    values: dict[str, List[np.ndarray]] = {layer: [] for layer in models}
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            feats = backbone(x)
            for layer, model in models.items():
                feat = _normalize_feature(feats[layer], bool(args.feature_normalize))
                _, loc = _feature_loss_and_map(model, feat, float(args.l1_weight), 0.0, 0.0)
                values[layer].append(loc.detach().flatten().cpu().numpy())
    return {
        layer: np.concatenate(parts).astype(np.float64) if parts else np.array([], dtype=np.float64)
        for layer, parts in values.items()
    }


def _make_stats(train_maps: dict[str, np.ndarray], score_norm: str, q_low: float, q_high: float) -> dict[str, LayerStats]:
    stats: dict[str, LayerStats] = {}
    for layer, vals in train_maps.items():
        if vals.size == 0:
            vals = np.array([0.0], dtype=np.float64)
        mean = float(vals.mean())
        std = float(vals.std())
        if std < 1e-8:
            std = 1.0
        qa = float(np.quantile(vals, float(q_low)))
        qb = float(np.quantile(vals, float(q_high)))
        if qb <= qa + 1e-12:
            qb = qa + 1.0
        stats[layer] = LayerStats(mode=score_norm, mean=mean, std=std, q_low=qa, q_high=qb)
    return stats


def _apply_score_norm(loc: torch.Tensor, stats: LayerStats) -> torch.Tensor:
    if stats.mode == "none":
        return loc
    if stats.mode == "zscore":
        return (loc - float(stats.mean)) / float(stats.std)
    if stats.mode == "quantile":
        return (loc - float(stats.q_low)) / max(1e-12, float(stats.q_high) - float(stats.q_low))
    raise ValueError(f"Unknown score norm: {stats.mode}")


def _aggregate_map_score(score_map: np.ndarray, mode: str, topk_frac: float) -> float:
    s = np.asarray(score_map, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return float("nan")
    if mode == "max":
        return float(s.max())
    if mode == "mean":
        return float(s.mean())
    if mode == "topk_mean":
        k = max(1, int(math.ceil(float(topk_frac) * s.size)))
        return float(np.partition(s, -k)[-k:].mean())
    raise ValueError(f"Unknown image aggregation mode: {mode}")


def _score_image(
    models: dict[str, nn.Module],
    backbone: nn.Module,
    image_tensor: torch.Tensor,
    stats: dict[str, LayerStats],
    device: torch.device,
    args: argparse.Namespace,
) -> np.ndarray:
    for model in models.values():
        model.eval()
    with torch.no_grad():
        x = image_tensor.unsqueeze(0).to(device)
        feats = backbone(x)
        maps = []
        for layer, model in models.items():
            feat = _normalize_feature(feats[layer], bool(args.feature_normalize))
            _, loc = _feature_loss_and_map(model, feat, float(args.l1_weight), 0.0, 0.0)
            loc = _apply_score_norm(loc, stats[layer]).unsqueeze(1)
            loc = F.interpolate(
                loc,
                size=(int(args.image_resize), int(args.image_resize)),
                mode="bilinear",
                align_corners=False,
            )
            maps.append(loc)
        merged = torch.stack(maps, dim=0).mean(dim=0).squeeze(0).squeeze(0)
        return merged.detach().cpu().numpy().astype(np.float32)


def _build_models(
    backbone: nn.Module,
    layers: List[str],
    sample: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, nn.Module], dict[str, tuple[int, int, int]]]:
    with torch.no_grad():
        feats = backbone(sample.to(device))
    models: dict[str, nn.Module] = {}
    shapes: dict[str, tuple[int, int, int]] = {}
    for layer in layers:
        shape = tuple(map(int, feats[layer].shape[1:]))
        shapes[layer] = shape
        model = _make_feature_model(args, shape).to(device)
        models[layer] = model
    return models, shapes


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

        if bool(args.eval_pixel_auc):
            mask = _load_mask(item.mask_path, (int(args.image_resize), int(args.image_resize)))
            if int(args.pixel_auc_size) > 0:
                heat_eval = _resize_array(heat.astype(np.float32), int(args.pixel_auc_size), Image.BILINEAR)
                mask_eval = _resize_array((mask * 255).astype(np.uint8), int(args.pixel_auc_size), Image.NEAREST) > 0
            else:
                heat_eval = heat
                mask_eval = mask > 0
            pixel_scores.append(heat_eval.astype(np.float32).reshape(-1))
            pixel_labels.append(mask_eval.astype(np.uint8).reshape(-1))

        if i % max(1, int(args.progress_every)) == 0:
            print(
                f" {category}/{layers_label}/{hard_label}/{score_norm}: evaluated {i}/{len(test_rows)}",
                flush=True,
            )

    y = np.array([r["label"] for r in image_results], dtype=np.int64)
    s = np.array([r["score"] for r in image_results], dtype=np.float64)
    image_auc = roc_auc(s, y)
    image_ap = pr_auc(s, y)
    pixel_auc = float("nan")
    if bool(args.eval_pixel_auc) and pixel_scores:
        pixel_auc = roc_auc(np.concatenate(pixel_scores), np.concatenate(pixel_labels))

    combo_label = f"{_safe_label(layers_label)}_hard{_safe_label(hard_label)}_{score_norm}"
    _write_csv(
        Path(args.output_dir) / f"results_{category}_{combo_label}.csv",
        image_results,
        ["category", "image", "defect_type", "label", "score", "mask_found", "feature_layers", "hard_loss", "score_norm"],
    )

    row = {
        "category": category,
        "feature_layers": layers_label,
        "hard_loss": hard_label,
        "score_norm": score_norm,
        "image_auc": float(image_auc),
        "image_ap": float(image_ap),
        "pixel_auc": float(pixel_auc),
        "n_test_images": int(len(test_rows)),
        "n_test_anomaly": int(y.sum()),
    }
    print(
        f"{category}/{layers_label}/hard={hard_label}/norm={score_norm}: "
        f"image_auc={image_auc:.4f} image_ap={image_ap:.4f} pixel_auc={pixel_auc:.4f}",
        flush=True,
    )
    return row


def _count_params(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = [p for p in params if p.requires_grad]
    return int(sum(p.numel() for p in params))


def _run_combo(
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
        f"{category}: layers={layers_label} hard={hard_label} "
        f"train_images={len(train_paths)} test_images={len(test_rows)} shapes={shapes}",
        flush=True,
    )
    _train_models(models, backbone, train_loader, device, args, hard_frac, hard_weight)
    train_maps = _collect_train_maps(models, backbone, train_loader, device, args)

    backbone_flops = estimate_forward_flops_per_sample(backbone, (3, int(args.image_resize), int(args.image_resize)))
    backbone_params = _count_params(backbone, trainable_only=False)
    lipnet_params = int(sum(model_size_metrics(m)["param_count"] for m in models.values()))
    lipnet_bytes = int(sum(model_size_metrics(m)["param_bytes"] for m in models.values()))
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


def _write_summary(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fieldnames = [
        "category",
        "feature_layers",
        "hard_loss",
        "score_norm",
        "image_auc",
        "image_ap",
        "pixel_auc",
        "n_train_images",
        "n_test_images",
        "n_test_anomaly",
        "image_resize",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-convolutional Feature-LIPNet sweep for MVTec AD.")
    parser.add_argument("--data_root", default="data/mvtec_ad")
    parser.add_argument("--categories", default="screw,carpet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps_if_available")
    parser.add_argument("--output_dir", default="runs/benchmark/mvtec_feature_lipnet_fcn_sweep_v1")
    parser.add_argument("--image_resize", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--backbone_weights", default="default", choices=["default", "none"])
    parser.add_argument("--feature_layer_sets", nargs="+", default=["layer3", "layer2", "layer2,layer3"])
    parser.add_argument("--score_norms", nargs="+", default=["none", "quantile"], choices=["none", "zscore", "quantile"])
    parser.add_argument("--hard_loss_settings", nargs="+", default=["0:0", "0.10:1.0"])
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
    parser.add_argument("--eval_pixel_auc", action="store_true")
    parser.add_argument("--pixel_auc_size", type=int, default=256)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_test_images", type=int, default=0)
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    mvtec_root = _find_mvtec_root(Path(args.data_root))
    categories = _category_names(mvtec_root, args.categories)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_sets = [_parse_layer_set(x) for x in args.feature_layer_sets]
    hard_settings = [_parse_hard_setting(x) for x in args.hard_loss_settings]
    print(
        f"MVTec Feature-LIPNet-FCN root={mvtec_root} categories={categories} "
        f"device={device} layer_sets={args.feature_layer_sets} hard={args.hard_loss_settings} norms={args.score_norms}",
        flush=True,
    )

    rows: list[dict] = []
    for category in categories:
        for layers in layer_sets:
            for hard_setting in hard_settings:
                rows.extend(_run_combo(category, mvtec_root, layers, hard_setting, args, device))

    _write_summary(out_dir / "summary.csv", rows)
    print(f"Saved: {out_dir / 'summary.csv'}")
    if rows:
        best = sorted(rows, key=lambda r: float(r["image_auc"]), reverse=True)[0]
        print(
            "Best image-AUC: "
            f"{best['category']} layers={best['feature_layers']} hard={best['hard_loss']} "
            f"norm={best['score_norm']} image_auc={float(best['image_auc']):.4f} "
            f"pixel_auc={float(best['pixel_auc']):.4f}"
        )


if __name__ == "__main__":
    main()
