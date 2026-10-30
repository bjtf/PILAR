#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from lipnet.metrics import roc_auc
from lipnet.one_class import LIPNetAutoencoder
from lipnet.efficiency import estimate_forward_flops_per_sample, model_size_metrics
from lipnet.utils import load_json, resolve_device, set_seed


def _infer_input_channels(ds_cfg: Dict, model_cfg: Dict | None = None) -> int:
    if isinstance(model_cfg, dict) and model_cfg.get("input_channels") is not None:
        return int(model_cfg["input_channels"])
    if ds_cfg.get("input_channels") is not None:
        return int(ds_cfg["input_channels"])
    name = str(ds_cfg.get("name", "mnist")).lower()
    if name in {"mnist", "fashion_mnist", "fashion-mnist"}:
        return 1
    if name in {"cifar10", "cifar-10"}:
        return 1 if bool(ds_cfg.get("grayscale", True)) else 3
    return 1


def _is_power_of_two(v: int) -> bool:
    n = int(v)
    return n > 0 and (n & (n - 1)) == 0


def _valid_pyramid_start(input_size: int, start_size: int) -> bool:
    if input_size <= 0 or start_size <= 0 or input_size % start_size != 0:
        return False
    return _is_power_of_two(input_size // start_size)


def _default_pyramid_start(input_size: int) -> int:
    for cand in (input_size // 4, input_size // 2, input_size):
        if _valid_pyramid_start(input_size, cand):
            return int(cand)
    for cand in range(input_size, 0, -1):
        if _valid_pyramid_start(input_size, cand):
            return int(cand)
    return int(input_size)


def _sanitize_decoder(decoder_cfg: Dict, input_size: int) -> Dict:
    dec = dict(decoder_cfg or {})
    if dec.get("type") != "pyramid":
        return dec
    start = int(dec.get("start_size", 7))
    if not _valid_pyramid_start(int(input_size), start):
        dec["start_size"] = _default_pyramid_start(int(input_size))
    return dec


def _build_transforms(cfg: Dict):
    ops = []
    if cfg.get("grayscale", True):
        ops.append(transforms.Grayscale())
    resize = cfg.get("resize", None)
    if resize is not None:
        ops.append(transforms.Resize((int(resize), int(resize))))
    ops.append(transforms.ToTensor())
    norm = cfg.get("normalize", None)
    if norm:
        ops.append(transforms.Normalize(mean=norm.get("mean", [0.1307]), std=norm.get("std", [0.3081])))
    return transforms.Compose(ops)


def _dataset_from_cfg(cfg: Dict):
    ds_cfg = cfg["dataset"]
    transform = _build_transforms(ds_cfg)
    data_dir = ds_cfg.get("data_dir", "./data")
    name = str(ds_cfg.get("name", "mnist")).lower()
    if name in {"mnist"}:
        cls = datasets.MNIST
    elif name in {"fashion_mnist", "fashion-mnist"}:
        cls = datasets.FashionMNIST
    elif name in {"cifar10", "cifar-10"}:
        cls = datasets.CIFAR10
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    download = bool(ds_cfg.get("download", True))
    train = cls(data_dir, train=True, download=download, transform=transform)
    test = cls(data_dir, train=False, download=download, transform=transform)
    return train, test


def _targets(ds) -> torch.Tensor:
    t = ds.targets
    if isinstance(t, list):
        return torch.tensor(t)
    return t.detach().clone() if isinstance(t, torch.Tensor) else torch.tensor(t)


def _filter_by_class(ds, class_id: int, invert: bool = False) -> Subset:
    t = _targets(ds)
    mask = t != int(class_id) if invert else t == int(class_id)
    idx = mask.nonzero(as_tuple=False).squeeze(1)
    return Subset(ds, idx)


def _filter_exact_class(ds, class_id: int) -> Subset:
    return _filter_by_class(ds, class_id, invert=False)


def _selected_trial(cfg: Dict) -> Dict:
    """Use the frozen architecture supplied with the paper."""
    trial = cfg.get("selected_trial")
    if not isinstance(trial, dict):
        raise ValueError("The configuration must contain selected_trial")
    return trial


def _parse_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _apply_inhibition_override(
    layers: List[Dict],
    mode: str,
    fixed_delta: float = 1.0,
    h: int = 1,
    radial_mu: float = 1.0,
    radial_sigma: float = 1.0,
) -> List[Dict]:
    if mode == "keep":
        return layers
    out = []
    for layer in layers:
        new_layer = dict(layer)
        if mode == "none":
            new_layer["inhibition"] = {"enabled": False}
        elif mode == "fixed":
            new_layer["inhibition"] = {
                "enabled": True,
                "mode": "fixed",
                "delta": float(fixed_delta),
                "h": int(h),
            }
        elif mode == "radial":
            new_layer["inhibition"] = {
                "enabled": True,
                "mode": "radial",
                "h": int(h),
                "radial_init_mu": float(radial_mu),
                "radial_init_sigma": float(radial_sigma),
            }
        else:
            raise ValueError(f"Unknown inhibition override: {mode}")
        out.append(new_layer)
    return out


def _apply_encoder_channels_override(layers: List[Dict], channels: List[int]) -> List[Dict]:
    if not channels:
        return layers
    out = []
    for i, layer in enumerate(layers):
        new_layer = dict(layer)
        new_layer["channels"] = int(channels[min(i, len(channels) - 1)])
        out.append(new_layer)
    return out


def _apply_encoder_layers_keep(layers: List[Dict], keep: int) -> List[Dict]:
    keep = int(keep)
    if keep <= 0:
        return layers
    return list(layers[:keep])


def _write_csv(path: Path, rows: Iterable[Dict], fieldnames: List[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)


def _train_reconstruction(model, loader, device, epochs: int, lr: float, loss_type: str, l1_weight: float) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    mse = torch.nn.MSELoss()
    l1 = torch.nn.L1Loss()
    for _ in range(int(epochs)):
        for x, _ in loader:
            x = x.to(device)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            opt.zero_grad()
            recon = model(x)
            loss = mse(recon, x)
            if loss_type == "mse_l1":
                loss = loss + float(l1_weight) * l1(recon, x)
            loss.backward()
            opt.step()
        print("|", end="", flush=True)


def _recon_errors(model, loader, device, loss_type: str, l1_weight: float) -> torch.Tensor:
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            recon = model(x)
            mse = ((recon - x) ** 2).mean(dim=[1, 2, 3])
            if loss_type == "mse_l1":
                l1 = torch.abs(recon - x).mean(dim=[1, 2, 3])
                mse = mse + float(l1_weight) * l1
            out.append(mse.cpu())
    return torch.cat(out) if out else torch.tensor([])


def _auc_from_scores(normal_scores: torch.Tensor, anomaly_scores: torch.Tensor) -> float:
    if normal_scores.numel() == 0 or anomaly_scores.numel() == 0:
        return float("nan")
    scores = torch.cat([normal_scores, anomaly_scores]).numpy()
    labels = np.concatenate([np.zeros(normal_scores.numel(), dtype=np.int64), np.ones(anomaly_scores.numel(), dtype=np.int64)])
    return roc_auc(scores, labels)


def _score_norm(scores: torch.Tensor, mu: float, std: float, enabled: bool) -> torch.Tensor:
    if not enabled:
        return scores
    return (scores - float(mu)) / (float(std) + 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruction-only no-prototype static-image benchmark.")
    parser.add_argument("--config", default="configs/benchmark_pyramid_fashion.json")
    parser.add_argument("--csv_out", default="runs/benchmark/oneclass_auc_per_class.csv")
    parser.add_argument("--pairwise_csv_out", default="runs/benchmark/oneclass_auc_pairwise.csv")
    parser.add_argument("--decoder_type", default="pure_pyramid", choices=["pure_pyramid", "pyramid"])
    parser.add_argument("--channels", default="16,8")
    parser.add_argument("--use_upsample", action="store_true")
    parser.add_argument("--epochs_mult", type=int, default=1)
    parser.add_argument("--loss_type", default="mse", choices=["mse", "mse_l1"])
    parser.add_argument("--l1_weight", type=float, default=0.2)
    parser.add_argument("--score_znorm", action="store_true")
    parser.add_argument("--recon_only", action="store_true")
    parser.add_argument("--latent_score_weight", type=float, default=0.0)
    parser.add_argument("--combo_sign_calibration", default="off")
    parser.add_argument("--inhibition_mode", default="keep", choices=["keep", "none", "fixed", "radial"])
    parser.add_argument("--inhibition_fixed_delta", type=float, default=1.0)
    parser.add_argument("--inhibition_h", type=int, default=1)
    parser.add_argument("--inhibition_radial_mu", type=float, default=1.0)
    parser.add_argument("--inhibition_radial_sigma", type=float, default=1.0)
    parser.add_argument("--encoder_channels_override", default="")
    parser.add_argument("--encoder_layers_keep", type=int, default=0)
    parser.add_argument(
        "--neg_class_ids",
        default="",
        help="Optional comma-separated anomaly classes. If omitted, uses classes 0-9, excluding the current normal class.",
    )
    parser.add_argument("--ablation_tag", default="")
    args = parser.parse_args()

    if not args.recon_only:
        raise SystemExit("Use --recon_only for the paper reconstruction protocol.")
    if float(args.latent_score_weight) != 0.0:
        raise SystemExit("--latent_score_weight must be 0.0 for reconstruction-only scoring.")
    if str(args.combo_sign_calibration) != "off":
        raise SystemExit("--combo_sign_calibration must be off in the strict no-prototype path.")

    outputs = [Path(args.csv_out), Path(args.pairwise_csv_out)]
    artifact_dir = Path(args.csv_out).parent / (Path(args.csv_out).stem + "_artifacts")
    if any(p.exists() for p in outputs) or artifact_dir.exists():
        raise FileExistsError("Outputs already exist. Choose new output paths; completed runs can be collected with the study runner --resume.")
    cfg = load_json(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "cuda_if_available"))
    trial = _selected_trial(cfg)

    model_cfg = dict(cfg["model"])
    model_cfg["layers_2d"] = trial["layers_2d"]
    model_cfg["decoder_hidden"] = trial.get("decoder_hidden", model_cfg.get("decoder_hidden", [128]))
    model_cfg["layers_2d"] = _apply_inhibition_override(
        model_cfg["layers_2d"],
        args.inhibition_mode,
        fixed_delta=args.inhibition_fixed_delta,
        h=args.inhibition_h,
        radial_mu=args.inhibition_radial_mu,
        radial_sigma=args.inhibition_radial_sigma,
    )
    model_cfg["layers_2d"] = _apply_encoder_channels_override(model_cfg["layers_2d"], _parse_ints(args.encoder_channels_override))
    model_cfg["layers_2d"] = _apply_encoder_layers_keep(model_cfg["layers_2d"], args.encoder_layers_keep)

    dec = dict(trial.get("decoder", cfg.get("decoder", model_cfg.get("decoder", {}))) or {})
    if args.decoder_type == "pure_pyramid":
        dec["type"] = "pure_pyramid"
        dec["channels"] = _parse_ints(args.channels)
        dec["use_upsample"] = bool(args.use_upsample)
    else:
        dec["type"] = "pyramid"

    train_full, test_ds = _dataset_from_cfg(cfg)
    sample_x, _ = train_full[0]
    input_channels = _infer_input_channels(cfg.get("dataset", {}), model_cfg)
    if sample_x.dim() == 3:
        input_channels = int(sample_x.shape[0])
    input_size = int(sample_x.shape[-1])
    model_cfg["input_channels"] = int(input_channels)
    model_cfg["input_size"] = int(input_size)
    dec = _sanitize_decoder(dec, input_size)

    ds_cfg = cfg["dataset"]
    batch_size = int(ds_cfg.get("batch_size", 128))
    class_ids = [int(c) for c in cfg.get("class_ids", list(range(10)))]
    neg_class_ids = _parse_ints(args.neg_class_ids) if str(args.neg_class_ids).strip() else list(range(10))
    base_epochs = int(cfg.get("training", {}).get("epochs", 20))
    epochs = base_epochs * int(args.epochs_mult)
    lr = float(cfg.get("training", {}).get("lr", 1e-3))
    dataset_name = str(ds_cfg.get("name", "unknown"))
    seed = int(cfg.get("seed", 42))
    if len(set(class_ids)) != len(class_ids) or any(c not in range(10) for c in class_ids) or not class_ids:
        raise ValueError("class_ids must be nonempty, unique and within 0..9")
    if len(set(neg_class_ids)) != len(neg_class_ids) or any(c not in range(10) for c in neg_class_ids):
        raise ValueError("neg_class_ids must be unique and within 0..9")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "run_config.json").write_text(json.dumps({"config": cfg, "arguments": vars(args), "resolved_model": model_cfg, "resolved_decoder": dec, "epochs": epochs, "device": str(device), "torch_version": str(torch.__version__), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")

    print(
        f"[run] dataset={dataset_name} seed={seed} device={device} "
        f"decoder={dec.get('type')} classes={','.join(map(str, class_ids))}"
    )

    rows = []
    pair_rows = []
    for class_id in class_ids:
        print(f"\nTraining class {class_id}")
        train_norm = _filter_by_class(train_full, class_id)
        train_loader = DataLoader(train_norm, batch_size=batch_size, shuffle=True)
        train_eval_loader = DataLoader(train_norm, batch_size=batch_size, shuffle=False)

        model = LIPNetAutoencoder(
            input_size=input_size,
            input_channels=input_channels,
            layers_2d=model_cfg["layers_2d"],
            decoder_hidden=model_cfg.get("decoder_hidden", [128]),
            decoder_cfg=dec,
        ).to(device)
        size_metrics = model_size_metrics(model)
        try:
            flops = estimate_forward_flops_per_sample(model, (input_channels, input_size, input_size))
            forward_flops = float(flops["flops_forward_total"])
        except Exception:
            forward_flops = float("nan")
        score_flops = float(input_channels * input_size * input_size * 3)
        infer_flops = float(forward_flops + score_flops) if forward_flops == forward_flops else float("nan")
        _train_reconstruction(model, train_loader, device, epochs, lr, args.loss_type, args.l1_weight)

        train_scores = _recon_errors(model, train_eval_loader, device, args.loss_type, args.l1_weight)
        mu = float(train_scores.mean().item()) if train_scores.numel() else 0.0
        std = float(train_scores.std().item()) if train_scores.numel() else 1.0

        normal_loader = DataLoader(_filter_exact_class(test_ds, class_id), batch_size=batch_size, shuffle=False)
        normal_scores_raw = _recon_errors(model, normal_loader, device, args.loss_type, args.l1_weight)
        normal_scores = _score_norm(normal_scores_raw, mu, std, args.score_znorm)

        score_rows = [{"test_index": int(i), "true_class": int(class_id), "label": 0, "score": float(v)} for i, v in zip(_filter_exact_class(test_ds, class_id).indices.tolist(), normal_scores.tolist())]
        anomaly_parts = []
        for neg_class in neg_class_ids:
            if int(neg_class) == int(class_id):
                continue
            neg_loader = DataLoader(_filter_exact_class(test_ds, int(neg_class)), batch_size=batch_size, shuffle=False)
            neg_scores_raw = _recon_errors(model, neg_loader, device, args.loss_type, args.l1_weight)
            neg_scores = _score_norm(neg_scores_raw, mu, std, args.score_znorm)
            anomaly_parts.append(neg_scores)
            score_rows.extend({"test_index": int(i), "true_class": int(neg_class), "label": 1, "score": float(v)} for i, v in zip(_filter_exact_class(test_ds, int(neg_class)).indices.tolist(), neg_scores.tolist()))
            pair_auc = _auc_from_scores(normal_scores, neg_scores)
            pair_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "ablation_tag": args.ablation_tag,
                    "pos_class": int(class_id),
                    "neg_class": int(neg_class),
                    "auc_recon": pair_auc,
                    "auc_combo": pair_auc,
                    "auc_latent": float("nan"),
                }
            )

        anomaly_scores = torch.cat(anomaly_parts) if anomaly_parts else torch.tensor([])
        auc_test = _auc_from_scores(normal_scores, anomaly_scores)
        rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "ablation_tag": args.ablation_tag,
                "class_id": int(class_id),
                "auc_test_recon": auc_test,
                "auc_test_combo": auc_test,
                "auc_test_latent": float("nan"),
                "train_recon_mu": mu,
                "train_recon_std": std,
                "n_train_normal": len(train_norm),
                "n_test_normal": int(normal_scores.numel()),
                "n_test_anomaly": int(anomaly_scores.numel()),
                "params": int(size_metrics["param_count"]),
                "param_bytes": int(size_metrics["param_bytes"]),
                "forward_flops": float(forward_flops),
                "infer_flops": float(infer_flops),
            }
        )
        _write_csv(artifact_dir / f"scores_class{class_id}.csv", score_rows, ["test_index", "true_class", "label", "score"])
        torch.save({"schema": "pilar-strict-inference-v1", "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "model_config": model_cfg, "decoder_config": dec, "config": cfg, "arguments": vars(args), "normal_class": int(class_id), "seed": seed, "epoch": epochs, "train_recon_mu": mu, "train_recon_std": std}, artifact_dir / f"model_class{class_id}.pt")
        print(f" class={class_id} auc={auc_test:.4f}")

    _write_csv(
        Path(args.csv_out),
        rows,
        [
            "dataset",
            "seed",
            "ablation_tag",
            "class_id",
            "auc_test_recon",
            "auc_test_combo",
            "auc_test_latent",
            "train_recon_mu",
            "train_recon_std",
            "n_train_normal",
            "n_test_normal",
            "n_test_anomaly",
            "params",
            "param_bytes",
            "forward_flops",
            "infer_flops",
        ],
    )
    _write_csv(
        Path(args.pairwise_csv_out),
        pair_rows,
        ["dataset", "seed", "ablation_tag", "pos_class", "neg_class", "auc_recon", "auc_combo", "auc_latent"],
    )


if __name__ == "__main__":
    main()
