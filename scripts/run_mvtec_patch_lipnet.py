#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
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
from torch.utils.data import DataLoader, Dataset

from lipnet.efficiency import estimate_forward_flops_per_sample, model_size_metrics
from lipnet.metrics import pr_auc, roc_auc
from lipnet.one_class import LIPNetAutoencoder
from lipnet.pro_metrics import aupro_at_fpr
from lipnet.utils import resolve_device, set_seed


MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


@dataclass(frozen=True)
class PatchRef:
    path: Path
    x: int
    y: int


@dataclass(frozen=True)
class TestImage:
    path: Path
    label: int
    defect_type: str
    mask_path: Path | None


def _parse_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _resize_base_image(img: Image.Image, image_resize: int) -> Image.Image:
    size = int(image_resize)
    if size <= 0:
        return img
    if img.size == (size, size):
        return img
    return img.resize((size, size), Image.BILINEAR)


def _load_rgb(path: Path, image_resize: int = 0) -> Image.Image:
    return _resize_base_image(Image.open(path).convert("RGB"), image_resize)


def _image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _patch_coords(width: int, height: int, patch_size: int, stride: int) -> List[tuple[int, int]]:
    if width <= patch_size:
        xs = [0]
    else:
        xs = list(range(0, max(1, width - patch_size + 1), stride))
        if xs[-1] != width - patch_size:
            xs.append(width - patch_size)
    if height <= patch_size:
        ys = [0]
    else:
        ys = list(range(0, max(1, height - patch_size + 1), stride))
        if ys[-1] != height - patch_size:
            ys.append(height - patch_size)
    return [(x, y) for y in ys for x in xs]


def _crop_patch(img: Image.Image, x: int, y: int, patch_size: int) -> Image.Image:
    width, height = img.size
    if width < patch_size or height < patch_size:
        scale = max(patch_size / max(1, width), patch_size / max(1, height))
        new_size = (max(patch_size, int(round(width * scale))), max(patch_size, int(round(height * scale))))
        img = img.resize(new_size, Image.BILINEAR)
        width, height = img.size
        x = min(x, width - patch_size)
        y = min(y, height - patch_size)
    return img.crop((x, y, x + patch_size, y + patch_size))


def _list_images(path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _find_mvtec_root(data_root: Path) -> Path:
    candidates = [
        data_root,
        data_root / "mvtec",
        data_root / "mvtec_ad",
        data_root / "MVTec",
        data_root / "MVTec_AD",
        data_root / "mvtec_anomaly_detection",
    ]
    for cand in candidates:
        if (cand / "bottle" / "train" / "good").exists() or any((cand / c / "train" / "good").exists() for c in MVTEC_CATEGORIES):
            return cand
    raise FileNotFoundError(
        "Could not find MVTec AD. Expected layout like "
        "data/mvtec_ad/bottle/train/good/*.png and data/mvtec_ad/bottle/test/<defect>/*.png."
    )


def _find_mask(category_root: Path, defect_type: str, image_path: Path) -> Path | None:
    if defect_type == "good":
        return None
    gt_dir = category_root / "ground_truth" / defect_type
    if not gt_dir.exists():
        return None
    stem = image_path.stem
    candidates = [
        gt_dir / f"{stem}_mask.png",
        gt_dir / f"{stem}.png",
        gt_dir / image_path.name,
        gt_dir / f"{stem}_mask{image_path.suffix}",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    matches = sorted(gt_dir.glob(f"{stem}*"))
    return matches[0] if matches else None


def _category_names(root: Path, selected: str) -> List[str]:
    if selected.strip().lower() == "all":
        return [c for c in MVTEC_CATEGORIES if (root / c / "train" / "good").exists()]
    out = [x.strip() for x in selected.split(",") if x.strip()]
    missing = [c for c in out if not (root / c / "train" / "good").exists()]
    if missing:
        raise FileNotFoundError(f"Missing MVTec categories under {root}: {missing}")
    return out


class MVTecPatchDataset(Dataset):
    def __init__(self, refs: List[PatchRef], patch_size: int, image_resize: int = 0) -> None:
        self.refs = list(refs)
        self.patch_size = int(patch_size)
        self.image_resize = int(image_resize)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ref = self.refs[idx]
        img = _load_rgb(ref.path, image_resize=self.image_resize)
        patch = _crop_patch(img, ref.x, ref.y, self.patch_size)
        return _image_to_tensor(patch)


def _make_train_refs(
    train_paths: List[Path],
    patch_size: int,
    stride: int,
    max_patches: int,
    seed: int,
    image_resize: int = 0,
) -> List[PatchRef]:
    refs: List[PatchRef] = []
    for path in train_paths:
        with _load_rgb(path, image_resize=int(image_resize)) as img:
            width, height = img.size
        for x, y in _patch_coords(width, height, patch_size, stride):
            refs.append(PatchRef(path=path, x=x, y=y))
    rng = random.Random(int(seed))
    rng.shuffle(refs)
    if max_patches > 0 and len(refs) > max_patches:
        refs = refs[:max_patches]
    return refs


def _make_model(args: argparse.Namespace) -> LIPNetAutoencoder:
    channels = _parse_ints(args.encoder_channels)
    rfs = _parse_ints(args.rf_schedule)
    overlaps = _parse_ints(args.overlap_schedule)
    if not (len(channels) == len(rfs) == len(overlaps)):
        raise ValueError("--encoder_channels, --rf_schedule, and --overlap_schedule must have the same length")

    layers = []
    for i, (ch, rf, ov) in enumerate(zip(channels, rfs, overlaps)):
        inhib = {
            "enabled": True,
            "mode": "fixed" if i == 0 else "radial",
            "delta": 1.0,
            "h": int(args.inhibition_h),
            "radial_init_mu": 1.0,
            "radial_init_sigma": 1.0,
        }
        layers.append(
            {
                "rf": int(rf),
                "overlap": int(ov),
                "activation": "relu",
                "channels": int(ch),
                "inhibition": inhib,
            }
        )

    dec = {
        "type": "pure_pyramid",
        "channels": _parse_ints(args.decoder_channels),
        "activation": "relu",
        "use_upsample": True,
        "upsample_mode": "bilinear",
    }
    if args.refine_decoder:
        dec["refine"] = {"type": "depthwise", "activation": "relu", "residual": True}

    return LIPNetAutoencoder(
        input_size=int(args.patch_size),
        input_channels=3,
        output_channels=3,
        layers_2d=layers,
        decoder_hidden=[],
        decoder_cfg=dec,
    )


def _loss_and_patch_scores(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor | None,
    l1_weight: float,
    topk_pixel_frac: float,
    topk_weight: float,
    edge_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target is None:
        target = x
    recon = model(x)
    residual = recon - target
    mse_map = residual.square().mean(dim=1)
    l1_map = residual.abs().mean(dim=1)
    pixel_map = mse_map + float(l1_weight) * l1_map
    if edge_weight > 0.0:
        edge_residual = _laplacian(recon) - _laplacian(target)
        edge_map = edge_residual.square().mean(dim=1)
        pixel_map = pixel_map + float(edge_weight) * edge_map
    mean_score = pixel_map.mean(dim=(1, 2))
    if topk_pixel_frac > 0.0 and topk_weight != 0.0:
        flat = pixel_map.flatten(1)
        k = max(1, int(math.ceil(float(topk_pixel_frac) * flat.shape[1])))
        topk_score = flat.topk(k, dim=1).values.mean(dim=1)
        score = mean_score + float(topk_weight) * topk_score
    else:
        score = mean_score
    loss = mean_score.mean()
    return loss, score


def _laplacian(x: torch.Tensor) -> torch.Tensor:
    c = int(x.shape[1])
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(c, 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=c)


def _corrupt_training_input(x: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    mode = str(args.train_corruption)
    if mode == "none":
        return x
    out = x.clone()
    b, c, h, w = out.shape
    prob = float(args.corruption_prob)
    if mode in {"noise", "cutout_noise"}:
        noise = torch.randn_like(out) * float(args.noise_std)
        apply = (torch.rand(b, 1, 1, 1, device=out.device) < prob).to(out.dtype)
        out = torch.clamp(out + apply * noise, 0.0, 1.0)
    if mode in {"cutout", "cutout_noise"}:
        count = max(1, int(args.cutout_count))
        min_frac = float(args.cutout_min_frac)
        max_frac = float(args.cutout_max_frac)
        for i in range(b):
            if float(torch.rand((), device=out.device).item()) >= prob:
                continue
            for _ in range(count):
                fh = float(torch.empty((), device=out.device).uniform_(min_frac, max_frac).item())
                fw = float(torch.empty((), device=out.device).uniform_(min_frac, max_frac).item())
                rh = max(2, min(h, int(round(fh * h))))
                rw = max(2, min(w, int(round(fw * w))))
                yy = int(torch.randint(0, max(1, h - rh + 1), (), device=out.device).item())
                xx = int(torch.randint(0, max(1, w - rw + 1), (), device=out.device).item())
                if str(args.cutout_fill) == "zero":
                    fill = torch.zeros((c, rh, rw), dtype=out.dtype, device=out.device)
                elif str(args.cutout_fill) == "mean":
                    fill = out[i].mean(dim=(1, 2), keepdim=True).expand(c, rh, rw)
                else:
                    fill = torch.rand((c, rh, rw), dtype=out.dtype, device=out.device)
                out[i, :, yy : yy + rh, xx : xx + rw] = fill
    return out


def _train(
    model: torch.nn.Module,
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
            target = x.to(device)
            x_in = _corrupt_training_input(target, args)
            opt.zero_grad(set_to_none=True)
            loss, _ = _loss_and_patch_scores(
                model,
                x_in,
                target=target,
                l1_weight=float(args.l1_weight),
                topk_pixel_frac=float(args.train_topk_pixel_frac),
                topk_weight=float(args.train_topk_weight),
                edge_weight=float(args.train_edge_weight),
            )
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(target.shape[0])
            n += int(target.shape[0])
        print(f" epoch={epoch}/{args.epochs} loss={total / max(1, n):.6f}", flush=True)


def _score_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> np.ndarray:
    model.eval()
    scores: List[np.ndarray] = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            _, s = _loss_and_patch_scores(
                model,
                x,
                target=None,
                l1_weight=float(args.l1_weight),
                topk_pixel_frac=float(args.score_topk_pixel_frac),
                topk_weight=float(args.score_topk_weight),
                edge_weight=float(args.score_edge_weight),
            )
            scores.append(s.cpu().numpy())
    return np.concatenate(scores) if scores else np.array([], dtype=np.float32)


def _score_patches(
    model: torch.nn.Module,
    patches: torch.Tensor,
    device: torch.device,
    batch_size: int,
    args: argparse.Namespace,
) -> np.ndarray:
    model.eval()
    scores: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, patches.shape[0], batch_size):
            x = patches[start : start + batch_size].to(device)
            _, s = _loss_and_patch_scores(
                model,
                x,
                target=None,
                l1_weight=float(args.l1_weight),
                topk_pixel_frac=float(args.score_topk_pixel_frac),
                topk_weight=float(args.score_topk_weight),
                edge_weight=float(args.score_edge_weight),
            )
            scores.append(s.cpu().numpy())
    return np.concatenate(scores) if scores else np.array([], dtype=np.float32)


def _score_patches_with_pixel_maps(
    model: torch.nn.Module,
    patches: torch.Tensor,
    device: torch.device,
    batch_size: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, List[np.ndarray]]:
    model.eval()
    scores: List[np.ndarray] = []
    maps: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, patches.shape[0], batch_size):
            x = patches[start : start + batch_size].to(device)
            recon = model(x)
            residual = recon - x
            mse_map = residual.square().mean(dim=1)
            l1_map = residual.abs().mean(dim=1)
            pixel_map = mse_map + float(args.l1_weight) * l1_map
            if float(args.score_edge_weight) > 0.0:
                edge_residual = _laplacian(recon) - _laplacian(x)
                edge_map = edge_residual.square().mean(dim=1)
                pixel_map = pixel_map + float(args.score_edge_weight) * edge_map
            mean_score = pixel_map.mean(dim=(1, 2))
            if float(args.score_topk_pixel_frac) > 0.0 and float(args.score_topk_weight) != 0.0:
                flat = pixel_map.flatten(1)
                k = max(1, int(math.ceil(float(args.score_topk_pixel_frac) * flat.shape[1])))
                topk_score = flat.topk(k, dim=1).values.mean(dim=1)
                score = mean_score + float(args.score_topk_weight) * topk_score
            else:
                score = mean_score
            scores.append(score.cpu().numpy())
            maps.extend([m for m in pixel_map.cpu().numpy()])
    return (np.concatenate(scores) if scores else np.array([], dtype=np.float32), maps)


def _image_patches(path: Path, patch_size: int, stride: int, image_resize: int) -> tuple[torch.Tensor, List[tuple[int, int]], tuple[int, int]]:
    img = _load_rgb(path, image_resize=image_resize)
    width, height = img.size
    coords = _patch_coords(width, height, patch_size, stride)
    patches = [_image_to_tensor(_crop_patch(img, x, y, patch_size)) for x, y in coords]
    return torch.stack(patches, dim=0), coords, (width, height)


def _aggregate_patch_scores(scores: np.ndarray, mode: str, topk_frac: float) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return float("nan")
    if mode == "max":
        return float(s.max())
    if mode == "mean":
        return float(s.mean())
    if mode == "topk_mean":
        k = max(1, int(math.ceil(float(topk_frac) * s.size)))
        return float(np.partition(s, -k)[-k:].mean())
    raise ValueError(f"Unknown aggregation mode: {mode}")


def _heatmap_from_patches(
    scores: np.ndarray,
    coords: List[tuple[int, int]],
    image_size: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    width, height = image_size
    heat = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    for score, (x, y) in zip(scores, coords):
        x2 = min(width, x + patch_size)
        y2 = min(height, y + patch_size)
        heat[y:y2, x:x2] += float(score)
        count[y:y2, x:x2] += 1.0
    return heat / np.maximum(count, 1.0)


def _heatmap_from_patch_maps(
    maps: List[np.ndarray],
    coords: List[tuple[int, int]],
    image_size: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    width, height = image_size
    heat = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    for patch_map, (x, y) in zip(maps, coords):
        x2 = min(width, x + patch_size)
        y2 = min(height, y + patch_size)
        ph = max(1, y2 - y)
        pw = max(1, x2 - x)
        arr = np.asarray(patch_map, dtype=np.float32)
        if arr.shape != (ph, pw):
            arr = _resize_array(arr.astype(np.float32), max(ph, pw), Image.BILINEAR)
            arr = arr[:ph, :pw]
            if arr.shape != (ph, pw):
                arr = np.asarray(Image.fromarray(np.asarray(patch_map, dtype=np.float32)).resize((pw, ph), Image.BILINEAR))
        heat[y:y2, x:x2] += arr
        count[y:y2, x:x2] += 1.0
    return heat / np.maximum(count, 1.0)


def _load_mask(mask_path: Path | None, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if mask_path is None:
        return np.zeros((height, width), dtype=np.uint8)
    mask = Image.open(mask_path).convert("L")
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def _resize_array(arr: np.ndarray, size: int, resample: int) -> np.ndarray:
    img = Image.fromarray(arr)
    img = img.resize((size, size), resample)
    return np.asarray(img)


def _test_images(category_root: Path) -> List[TestImage]:
    test_root = category_root / "test"
    rows: List[TestImage] = []
    for defect_dir in sorted([p for p in test_root.iterdir() if p.is_dir()]):
        defect_type = defect_dir.name
        label = 0 if defect_type == "good" else 1
        for path in _list_images(defect_dir):
            rows.append(
                TestImage(
                    path=path,
                    label=label,
                    defect_type=defect_type,
                    mask_path=_find_mask(category_root, defect_type, path),
                )
            )
    return rows


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)


def _run_category(category: str, mvtec_root: Path, args: argparse.Namespace, device: torch.device) -> dict:
    category_root = mvtec_root / category
    train_paths = _list_images(category_root / "train" / "good")
    test_rows = _test_images(category_root)
    if not train_paths:
        raise RuntimeError(f"{category}: no train/good images found")
    if not test_rows:
        raise RuntimeError(f"{category}: no test images found")

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

    model = _make_model(args).to(device)
    size = model_size_metrics(model)
    flops = estimate_forward_flops_per_sample(model, (3, int(args.patch_size), int(args.patch_size)))
    _train(model, loader, device, args)
    train_scores = _score_loader(model, loader, device, args)
    train_mu = float(train_scores.mean()) if train_scores.size else 0.0
    train_sd = float(train_scores.std()) if train_scores.size else 1.0
    if train_sd < 1e-8:
        train_sd = 1.0

    image_results = []
    pixel_scores: List[np.ndarray] = []
    pixel_labels: List[np.ndarray] = []
    pro_heat_maps: List[np.ndarray] = []
    pro_mask_maps: List[np.ndarray] = []
    compute_maps = bool(args.eval_pixel_auc) or bool(args.eval_aupro)
    for i, item in enumerate(test_rows, start=1):
        patches, coords, image_size = _image_patches(
            item.path,
            int(args.patch_size),
            int(args.test_patch_stride),
            int(args.image_resize),
        )
        patch_maps: List[np.ndarray] | None = None
        if compute_maps and str(args.pixel_heatmap_mode) == "residual":
            patch_scores, patch_maps = _score_patches_with_pixel_maps(model, patches, device, int(args.batch_size), args)
        else:
            patch_scores = _score_patches(model, patches, device, int(args.batch_size), args)
        if args.score_znorm:
            patch_scores = (patch_scores - train_mu) / train_sd
        image_score = _aggregate_patch_scores(patch_scores, args.image_agg, float(args.image_topk_frac))
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

        if compute_maps:
            if patch_maps is not None:
                heat = _heatmap_from_patch_maps(patch_maps, coords, image_size, int(args.patch_size))
            else:
                heat = _heatmap_from_patches(patch_scores, coords, image_size, int(args.patch_size))
            mask = _load_mask(item.mask_path, image_size)
            if args.eval_pixel_auc:
                heat_px = heat
                mask_px = mask
                if int(args.pixel_auc_size) > 0:
                    heat_px = _resize_array(heat_px.astype(np.float32), int(args.pixel_auc_size), Image.BILINEAR)
                    mask_px = _resize_array((mask_px * 255).astype(np.uint8), int(args.pixel_auc_size), Image.NEAREST) > 0
                pixel_scores.append(heat_px.astype(np.float32).reshape(-1))
                pixel_labels.append(mask_px.astype(np.uint8).reshape(-1))
            if args.eval_aupro:
                heat_pro = heat
                mask_pro = mask
                if int(args.pro_eval_size) > 0:
                    heat_pro = _resize_array(heat_pro.astype(np.float32), int(args.pro_eval_size), Image.BILINEAR)
                    mask_pro = _resize_array((mask_pro * 255).astype(np.uint8), int(args.pro_eval_size), Image.NEAREST) > 0
                pro_heat_maps.append(heat_pro.astype(np.float32))
                pro_mask_maps.append(mask_pro.astype(bool))

        if i % max(1, int(args.progress_every)) == 0:
            print(f" {category}: evaluated {i}/{len(test_rows)}", flush=True)

    y = np.array([r["label"] for r in image_results], dtype=np.int64)
    s = np.array([r["score"] for r in image_results], dtype=np.float64)
    image_auc = roc_auc(s, y)
    image_ap = pr_auc(s, y)
    avg_test_patches = float(np.mean([float(r["n_patches"]) for r in image_results])) if image_results else float("nan")
    pixel_auc = float("nan")
    if args.eval_pixel_auc and pixel_scores:
        px_s = np.concatenate(pixel_scores)
        px_y = np.concatenate(pixel_labels)
        pixel_auc = roc_auc(px_s, px_y)
    pro_info = {
        "aupro": float("nan"),
        "n_regions": 0,
        "n_normal_pixels": 0,
        "fpr_limit": float(args.fpr_limit),
        "num_thresholds": int(args.num_thresholds),
    }
    raw_curve: list[dict] = []
    clipped_curve: list[dict] = []
    if args.eval_aupro and pro_heat_maps:
        pro_info, raw_curve, clipped_curve = aupro_at_fpr(
            pro_heat_maps,
            pro_mask_maps,
            fpr_limit=float(args.fpr_limit),
            num_thresholds=int(args.num_thresholds),
            connectivity=int(args.connectivity),
        )

    cat_dir = Path(args.output_dir)
    _write_csv(
        cat_dir / f"results_{category}.csv",
        image_results,
        ["category", "image", "defect_type", "label", "score", "n_patches", "mask_found"],
    )
    if args.eval_aupro:
        _write_csv(
            cat_dir / f"pro_curve_raw_{category}.csv",
            raw_curve,
            ["threshold", "fpr", "pro"],
        )
        _write_csv(
            cat_dir / f"pro_curve_clipped_{category}.csv",
            clipped_curve,
            ["fpr", "pro"],
        )

    row = {
        "category": category,
        "image_auc": float(image_auc),
        "image_ap": float(image_ap),
        "pixel_auc": float(pixel_auc),
        "aupro_30": float(pro_info["aupro"]),
        "aupro_30_percent": float(pro_info["aupro"]) * 100.0,
        "fpr_limit": float(pro_info["fpr_limit"]),
        "num_thresholds": int(pro_info["num_thresholds"]),
        "n_pro_regions": int(pro_info["n_regions"]),
        "n_normal_pixels": int(pro_info["n_normal_pixels"]),
        "pro_eval_size": int(args.pro_eval_size),
        "connectivity": int(args.connectivity),
        "pixel_heatmap_mode": str(args.pixel_heatmap_mode),
        "n_train_images": int(len(train_paths)),
        "n_train_patches": int(len(refs)),
        "n_test_images": int(len(test_rows)),
        "n_test_anomaly": int(y.sum()),
        "params": int(size["param_count"]),
        "param_bytes": int(size["param_bytes"]),
        "forward_flops_per_patch": float(flops["flops_forward_total"]),
        "score_flops_per_patch": float(3 * int(args.patch_size) * int(args.patch_size) * 6),
        "avg_test_patches": float(avg_test_patches),
        "approx_forward_flops_per_image": float(avg_test_patches * float(flops["flops_forward_total"])),
        "approx_infer_flops_per_image": float(
            avg_test_patches * (float(flops["flops_forward_total"]) + float(3 * int(args.patch_size) * int(args.patch_size) * 6))
        ),
        "train_score_mu": float(train_mu),
        "train_score_sd": float(train_sd),
        "train_corruption": str(args.train_corruption),
        "train_edge_weight": float(args.train_edge_weight),
        "score_edge_weight": float(args.score_edge_weight),
    }
    print(
        f"{category}: image_auc={image_auc:.4f} image_ap={image_ap:.4f} "
        f"pixel_auc={pixel_auc:.4f} aupro@{float(args.fpr_limit):.2f}={float(pro_info['aupro']):.4f} "
        f"params={row['params']}",
        flush=True,
    )
    return row


def _finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray([float(x) for x in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _macro_row(rows: List[dict]) -> dict:
    fieldnames = _summary_fieldnames()
    out = {name: "" for name in fieldnames}
    out["category"] = "macro_mean"
    if not rows:
        return out
    for field in [
        "image_auc",
        "image_ap",
        "pixel_auc",
        "aupro_30",
        "aupro_30_percent",
        "params",
        "param_bytes",
        "forward_flops_per_patch",
        "score_flops_per_patch",
        "avg_test_patches",
        "approx_forward_flops_per_image",
        "approx_infer_flops_per_image",
    ]:
        out[field] = _finite_mean(float(r[field]) for r in rows)
    for field in [
        "fpr_limit",
        "num_thresholds",
        "pro_eval_size",
        "connectivity",
        "pixel_heatmap_mode",
        "train_corruption",
        "train_edge_weight",
        "score_edge_weight",
    ]:
        out[field] = rows[0].get(field, "")
    return out


def _summary_fieldnames() -> List[str]:
    return [
        "category",
        "image_auc",
        "image_ap",
        "pixel_auc",
        "aupro_30",
        "aupro_30_percent",
        "fpr_limit",
        "num_thresholds",
        "n_pro_regions",
        "n_normal_pixels",
        "pro_eval_size",
        "connectivity",
        "pixel_heatmap_mode",
        "n_train_images",
        "n_train_patches",
        "n_test_images",
        "n_test_anomaly",
        "params",
        "param_bytes",
        "forward_flops_per_patch",
        "score_flops_per_patch",
        "avg_test_patches",
        "approx_forward_flops_per_image",
        "approx_infer_flops_per_image",
        "train_score_mu",
        "train_score_sd",
        "train_corruption",
        "train_edge_weight",
        "score_edge_weight",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch-based LIPNet autoencoder benchmark for MVTec AD.")
    parser.add_argument("--data_root", default="data/mvtec_ad")
    parser.add_argument("--categories", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps_if_available")
    parser.add_argument("--output_dir", default="runs/benchmark/mvtec_patch_lipnet_v1")
    parser.add_argument(
        "--image_resize",
        type=int,
        default=0,
        help="Optional square resize before extracting patches. Use 0 to keep original image size.",
    )
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--train_patch_stride", type=int, default=64)
    parser.add_argument("--test_patch_stride", type=int, default=32)
    parser.add_argument("--max_train_patches", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--encoder_channels", default="3,8,8")
    parser.add_argument("--rf_schedule", default="5,5,3")
    parser.add_argument("--overlap_schedule", default="3,3,1")
    parser.add_argument("--inhibition_h", type=int, default=1)
    parser.add_argument("--decoder_channels", default="32,16,8")
    parser.add_argument("--refine_decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--l1_weight", type=float, default=0.2)
    parser.add_argument("--train_topk_pixel_frac", type=float, default=0.0)
    parser.add_argument("--train_topk_weight", type=float, default=0.0)
    parser.add_argument("--train_edge_weight", type=float, default=0.0)
    parser.add_argument("--train_corruption", default="none", choices=["none", "cutout", "noise", "cutout_noise"])
    parser.add_argument("--corruption_prob", type=float, default=0.75)
    parser.add_argument("--cutout_count", type=int, default=2)
    parser.add_argument("--cutout_min_frac", type=float, default=0.08)
    parser.add_argument("--cutout_max_frac", type=float, default=0.28)
    parser.add_argument("--cutout_fill", default="random", choices=["random", "zero", "mean"])
    parser.add_argument("--noise_std", type=float, default=0.08)
    parser.add_argument("--score_topk_pixel_frac", type=float, default=0.01)
    parser.add_argument("--score_topk_weight", type=float, default=1.0)
    parser.add_argument("--score_edge_weight", type=float, default=0.0)
    parser.add_argument("--score_znorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image_agg", default="topk_mean", choices=["topk_mean", "max", "mean"])
    parser.add_argument("--image_topk_frac", type=float, default=0.05)
    parser.add_argument("--eval_pixel_auc", action="store_true")
    parser.add_argument("--eval_aupro", action="store_true")
    parser.add_argument(
        "--pixel_heatmap_mode",
        default="patch_score",
        choices=["patch_score", "residual"],
        help="Use scalar patch scores or stitched reconstruction residual maps for pixel-AUROC.",
    )
    parser.add_argument("--pixel_auc_size", type=int, default=256)
    parser.add_argument("--pro_eval_size", type=int, default=256)
    parser.add_argument("--fpr_limit", type=float, default=0.30)
    parser.add_argument("--num_thresholds", type=int, default=400)
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--progress_every", type=int, default=25)
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    mvtec_root = _find_mvtec_root(Path(args.data_root))
    categories = _category_names(mvtec_root, args.categories)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"MVTec Patch-LIPNet root={mvtec_root} categories={categories} device={device}")

    rows = []
    for category in categories:
        rows.append(_run_category(category, mvtec_root, args, device))

    _write_csv(
        out_dir / "summary.csv",
        rows,
        _summary_fieldnames(),
    )
    if args.eval_aupro:
        _write_csv(out_dir / "summary_aupro.csv", rows + [_macro_row(rows)], _summary_fieldnames())

    image_aucs = np.array([r["image_auc"] for r in rows], dtype=np.float64)
    pixel_aucs = np.array([r["pixel_auc"] for r in rows], dtype=np.float64)
    print("Summary:")
    print(f" mean_image_auc={np.nanmean(image_aucs):.4f}")
    if np.isfinite(pixel_aucs).any():
        print(f" mean_pixel_auc={np.nanmean(pixel_aucs):.4f}")
    print(f" saved={out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
