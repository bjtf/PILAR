#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import hashlib
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
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from lipnet.efficiency import estimate_forward_flops_per_sample, model_size_metrics
from lipnet.metrics import pr_auc, roc_auc
from lipnet.utils import resolve_device, set_seed
from scripts.run_mvtec_feature_lipnet import _make_feature_model
from scripts.run_mvtec_feature_lipnet_fcn_sweep import (
    LayerStats,
    ResNetMultiFeatureExtractor,
    _apply_score_norm,
    _feature_loss_and_map,
    _make_stats,
    _normalize_feature,
    _parse_hard_setting,
    _parse_layer_set,
)


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]

DATASET_CLASSES = {
    "cifar10": CIFAR10_CLASSES,
    "fashion_mnist": FASHION_MNIST_CLASSES,
}


@dataclass(frozen=True)
class ClassSplit:
    class_id: int
    class_name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_labels: np.ndarray


def _parse_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _targets(ds) -> np.ndarray:
    vals = getattr(ds, "targets")
    if isinstance(vals, torch.Tensor):
        return vals.detach().cpu().numpy().astype(np.int64)
    return np.asarray(vals, dtype=np.int64)


def _make_transform(image_resize: int, grayscale_to_rgb: bool = False) -> transforms.Compose:
    steps: list = [transforms.Resize((int(image_resize), int(image_resize)))]
    if bool(grayscale_to_rgb):
        steps.append(transforms.Grayscale(num_output_channels=3))
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


def _load_cifar10(data_root: Path, image_resize: int, download: bool) -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    tfm = _make_transform(image_resize)
    train = datasets.CIFAR10(str(data_root), train=True, transform=tfm, download=bool(download))
    test = datasets.CIFAR10(str(data_root), train=False, transform=tfm, download=bool(download))
    return train, test


def _load_fashion_mnist(
    data_root: Path,
    image_resize: int,
    download: bool,
) -> tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    tfm = _make_transform(image_resize, grayscale_to_rgb=True)
    train = datasets.FashionMNIST(str(data_root), train=True, transform=tfm, download=bool(download))
    test = datasets.FashionMNIST(str(data_root), train=False, transform=tfm, download=bool(download))
    return train, test


def _load_dataset(
    name: str,
    data_root: Path,
    image_resize: int,
    download: bool,
):
    if name == "cifar10":
        return _load_cifar10(data_root, image_resize, download)
    if name == "fashion_mnist":
        return _load_fashion_mnist(data_root, image_resize, download)
    raise ValueError(f"Unsupported dataset: {name}")


def _build_split(
    train_ds,
    test_ds,
    class_id: int,
    class_names: list[str],
    max_train: int,
    max_test: int,
    rng: np.random.Generator,
) -> ClassSplit:
    train_targets = _targets(train_ds)
    test_targets = _targets(test_ds)
    train_idx = np.flatnonzero(train_targets == int(class_id))
    test_idx = np.arange(len(test_targets), dtype=np.int64)
    if int(max_train) > 0 and len(train_idx) > int(max_train):
        train_idx = np.sort(rng.choice(train_idx, size=int(max_train), replace=False))
    if int(max_test) > 0 and len(test_idx) > int(max_test):
        test_idx = np.sort(rng.choice(test_idx, size=int(max_test), replace=False))
    labels = (test_targets[test_idx] != int(class_id)).astype(np.int64)
    return ClassSplit(
        class_id=int(class_id),
        class_name=class_names[int(class_id)],
        train_idx=train_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        test_labels=labels,
    )


def _count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


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
        models[layer] = _make_feature_model(args, shape).to(device)
    return models, shapes


def _train_models(
    models: dict[str, nn.Module],
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    hard_frac: float,
    hard_weight: float,
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
        for x, _ in loader:
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
                    train_topk_feature_frac=float(hard_frac),
                    train_topk_feature_weight=float(hard_weight),
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
    values: dict[str, list[np.ndarray]] = {layer: [] for layer in models}
    with torch.no_grad():
        for x, _ in loader:
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


def _batch_score(
    models: dict[str, nn.Module],
    backbone: nn.Module,
    x: torch.Tensor,
    stats: dict[str, LayerStats],
    device: torch.device,
    args: argparse.Namespace,
) -> np.ndarray:
    x = x.to(device)
    with torch.no_grad():
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
        merged = torch.stack(maps, dim=0).mean(dim=0).squeeze(1)
        flat = merged.flatten(1)
        if args.image_agg == "max":
            score = flat.max(dim=1).values
        elif args.image_agg == "mean":
            score = flat.mean(dim=1)
        elif args.image_agg == "topk_mean":
            k = max(1, int(math.ceil(float(args.image_topk_frac) * flat.shape[1])))
            score = flat.topk(k, dim=1).values.mean(dim=1)
        else:
            raise ValueError(f"Unknown image aggregation: {args.image_agg}")
    return score.detach().cpu().numpy().astype(np.float64)


def _evaluate(
    split: ClassSplit,
    models: dict[str, nn.Module],
    backbone: nn.Module,
    test_loader: DataLoader,
    train_maps: dict[str, np.ndarray],
    score_norm: str,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    stats = _make_stats(train_maps, score_norm, float(args.quantile_low), float(args.quantile_high))
    for model in models.values():
        model.eval()

    scores: list[np.ndarray] = []
    true_labels: list[np.ndarray] = []
    true_classes: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    for x, class_labels, indices in test_loader:
        scores.append(_batch_score(models, backbone, x, stats, device, args))
        class_np = class_labels.detach().cpu().numpy().astype(np.int64)
        true_classes.append(class_np)
        true_labels.append((class_np != int(split.class_id)).astype(np.int64))
        sample_indices.append(indices.detach().cpu().numpy().astype(np.int64))

    s = np.concatenate(scores)
    y = np.concatenate(true_labels)
    cls = np.concatenate(true_classes)
    idx = np.concatenate(sample_indices)
    auc = roc_auc(s, y)
    ap = pr_auc(s, y)
    oracle_auc = max(float(auc), 1.0 - float(auc)) if not math.isnan(float(auc)) else float("nan")
    inverted = int(float(auc) < 0.5) if not math.isnan(float(auc)) else 0
    rows = [
        {
            "normal_class": int(split.class_id),
            "normal_name": split.class_name,
            "test_index": int(i),
            "true_class": int(c),
            "label": int(label),
            "score": float(score),
            "score_norm": str(score_norm),
        }
        for i, c, label, score in zip(idx, cls, y, s)
    ]
    summary = {
        "normal_class": int(split.class_id),
        "normal_name": split.class_name,
        "score_norm": str(score_norm),
        "auc": float(auc),
        "ap": float(ap),
        "oracle_auc": float(oracle_auc),
        "inverted": int(inverted),
        "n_train": int(len(split.train_idx)),
        "n_test": int(len(y)),
        "n_anomaly": int(y.sum()),
    }
    return summary, rows


class IndexedSubset(Dataset):
    def __init__(self, subset: Subset) -> None:
        self.subset = subset

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        x, y = self.subset[idx]
        return x, int(y), int(self.subset.indices[idx])


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)


def _run_one(
    train_ds,
    test_ds,
    class_id: int,
    seed: int,
    layers: List[str],
    hard_setting: tuple[float, float],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict]:
    rng = np.random.default_rng(int(seed) + 1009 * int(class_id))
    class_names = DATASET_CLASSES[str(args.dataset)]
    split = _build_split(
        train_ds,
        test_ds,
        class_id,
        class_names,
        int(args.max_train_images),
        int(args.max_test_images),
        rng,
    )

    train_loader = DataLoader(
        Subset(train_ds, split.train_idx.tolist()),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=False,
    )
    test_loader = DataLoader(
        IndexedSubset(Subset(test_ds, split.test_idx.tolist())),
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
    )
    sample = next(iter(train_loader))[0][:1]
    backbone = ResNetMultiFeatureExtractor(args.backbone, layers, args.backbone_weights).to(device).eval()
    models, shapes = _build_models(backbone, layers, sample, device, args)

    hard_frac, hard_weight = hard_setting
    layers_label = ",".join(layers)
    hard_label = f"{hard_frac:g}:{hard_weight:g}"
    print(
        f"seed={seed} class={class_id} {split.class_name} layers={layers_label} hard={hard_label} "
        f"train={len(split.train_idx)} test={len(split.test_idx)} shapes={shapes}",
        flush=True,
    )
    _train_models(models, backbone, train_loader, device, args, hard_frac, hard_weight)
    train_maps = _collect_train_maps(models, backbone, train_loader, device, args)

    backbone_flops = estimate_forward_flops_per_sample(backbone, (3, int(args.image_resize), int(args.image_resize)))
    backbone_params = _count_params(backbone)
    lipnet_params = int(sum(model_size_metrics(m)["param_count"] for m in models.values()))
    lipnet_bytes = int(sum(model_size_metrics(m)["param_bytes"] for m in models.values()))
    lipnet_flops = float(
        sum(estimate_forward_flops_per_sample(model, shapes[layer])["flops_forward_total"] for layer, model in models.items())
    )

    summaries: list[dict] = []
    for score_norm in args.score_norms:
        summary, score_rows = _evaluate(split, models, backbone, test_loader, train_maps, str(score_norm), device, args)
        summary.update(
            {
                "dataset": str(args.dataset),
                "seed": int(seed),
                "feature_layers": layers_label,
                "hard_loss": hard_label,
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
        summaries.append(summary)
        stats = _make_stats(train_maps, str(score_norm), float(args.quantile_low), float(args.quantile_high))
        torch.save({"schema": "pilar-feature-classification-inference-v1", "config": vars(args), "seed": int(seed), "normal_class": int(class_id), "epoch": int(args.epochs), "backbone": {k: v.detach().cpu() for k, v in backbone.state_dict().items()}, "models": {layer: {k: v.detach().cpu() for k, v in model.state_dict().items()} for layer, model in models.items()}, "feature_shapes": shapes, "stats": {layer: vars(value) for layer, value in stats.items()}, "train_indices": split.train_idx.tolist(), "test_indices": split.test_idx.tolist()}, Path(args.output_dir) / f"model_seed{seed}_class{class_id}_{score_norm}.pt")
        label = f"seed{seed}_class{class_id}_{str(score_norm)}"
        _write_csv(
            Path(args.output_dir) / f"scores_{label}.csv",
            score_rows,
            ["normal_class", "normal_name", "test_index", "true_class", "label", "score", "score_norm"],
        )
        print(
            f" seed={seed} class={class_id} norm={score_norm}: "
            f"auc={summary['auc']:.4f} ap={summary['ap']:.4f} "
            f"oracle={summary['oracle_auc']:.4f} inverted={summary['inverted']}",
            flush=True,
        )
    return summaries


def _summarize(rows: list[dict], expected_classes=None, expected_seeds=None) -> list[dict]:
    """Average classes within a seed; SD uses independent seed macros."""
    expected_classes = set(range(10) if expected_classes is None else expected_classes)
    expected_seeds = set(range(42, 47) if expected_seeds is None else expected_seeds)
    expected = {(seed, cls) for seed in expected_seeds for cls in expected_classes}
    out = []
    keys = sorted({(r["feature_layers"], r["hard_loss"], r["score_norm"]) for r in rows})
    if not rows:
        raise ValueError("No feature results to summarize")
    for layers, hard, norm in keys:
        sub = [r for r in rows if (r["feature_layers"], r["hard_loss"], r["score_norm"]) == (layers, hard, norm)]
        pairs = [(int(r["seed"]), int(r["normal_class"])) for r in sub]
        if len(pairs) != len(set(pairs)) or set(pairs) != expected:
            raise ValueError(f"Incomplete/duplicate class--seed matrix for {layers}/{hard}/{norm}")
        for metric in ("auc", "ap", "oracle_auc"):
            vals = np.asarray([float(r[metric]) for r in sub])
            if not np.isfinite(vals).all():
                raise ValueError(f"Non-finite {metric} in feature results")
            macros = np.asarray([np.mean([float(r[metric]) for r in sub if int(r["seed"]) == seed]) for seed in sorted(expected_seeds)])
            out.append({"feature_layers": layers, "hard_loss": hard, "score_norm": norm,
                        "metric": metric, "mean": float(macros.mean()),
                        "std": float(macros.std(ddof=1)) if len(macros) > 1 else float("nan"),
                        "n": len(macros), "n_classes": len(expected_classes),
                        "n_inverted": sum(int(r["inverted"]) for r in sub)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="One-vs-rest Feature-LIPNet-FCN benchmark.")
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASET_CLASSES))
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_dir", default="runs/benchmark/cifar10_feature_lipnet_fcn_layer23_hard_v1")
    parser.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--device", default="mps_if_available")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--image_resize", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
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
    parser.add_argument("--inhibition_h", type=int, default=1)
    parser.add_argument("--refine_decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--l1_weight", type=float, default=0.0)
    parser.add_argument("--quantile_low", type=float, default=0.90)
    parser.add_argument("--quantile_high", type=float, default=0.995)
    parser.add_argument("--image_agg", default="topk_mean", choices=["topk_mean", "max", "mean"])
    parser.add_argument("--image_topk_frac", type=float, default=0.01)
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_test_images", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Choose a new --output_dir.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    device = resolve_device(args.device)
    (out_dir / "run_metadata.json").write_text(json.dumps({"torch_version": str(torch.__version__), "resolved_device": str(device), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")
    classes = _parse_ints(args.classes)
    seeds = _parse_ints(args.seeds)
    if not classes or len(classes) != len(set(classes)) or any(c not in range(10) for c in classes):
        raise ValueError("classes must be nonempty, unique and within 0..9")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be nonempty and unique")
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.image_resize) <= 0 or not 0 < args.image_topk_frac <= 1:
        raise ValueError("epochs/batches/resolution must be positive and image_topk_frac in (0,1]")
    layers = _parse_layer_set(args.feature_layers)
    hard_setting = _parse_hard_setting(args.hard_loss)
    train_ds, test_ds = _load_dataset(str(args.dataset), Path(args.data_root), int(args.image_resize), bool(args.download))
    print(
        f"{args.dataset} Feature-LIPNet-FCN classes={classes} seeds={seeds} device={device} "
        f"layers={layers} hard={args.hard_loss} norms={args.score_norms}",
        flush=True,
    )

    all_rows: list[dict] = []
    for seed in seeds:
        set_seed(int(seed))
        for class_id in classes:
            all_rows.extend(_run_one(train_ds, test_ds, int(class_id), int(seed), layers, hard_setting, args, device))

    fieldnames = [
        "dataset",
        "seed",
        "normal_class",
        "normal_name",
        "feature_layers",
        "hard_loss",
        "score_norm",
        "auc",
        "ap",
        "oracle_auc",
        "inverted",
        "n_train",
        "n_test",
        "n_anomaly",
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
    _write_csv(out_dir / "results_long.csv", all_rows, fieldnames)
    summary = _summarize(all_rows, classes, seeds)
    _write_csv(
        out_dir / "summary.csv",
        summary,
        ["feature_layers", "hard_loss", "score_norm", "metric", "mean", "std", "n", "n_classes", "n_inverted"],
    )
    print(f"Saved: {out_dir / 'results_long.csv'}")
    print(f"Saved: {out_dir / 'summary.csv'}")
    for row in summary:
        if row["metric"] == "auc":
            print(
                f"Summary {row['feature_layers']}/hard={row['hard_loss']}/norm={row['score_norm']}: "
                f"mean_auc={row['mean']:.4f} std={row['std']:.4f} n={row['n']} inverted={row['n_inverted']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
