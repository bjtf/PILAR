from __future__ import annotations

import math

import numpy as np


def connected_regions(mask: np.ndarray, connectivity: int) -> list[tuple[np.ndarray, np.ndarray]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got shape={mask.shape}")
    if not mask.any():
        return []

    if int(connectivity) == 4:
        neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    elif int(connectivity) == 8:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        raise ValueError("connectivity must be 4 or 8")

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    regions: list[tuple[np.ndarray, np.ndarray]] = []
    start_ys, start_xs = np.nonzero(mask)

    for sy, sx in zip(start_ys.tolist(), start_xs.tolist()):
        if visited[sy, sx]:
            continue
        stack = [(sy, sx)]
        visited[sy, sx] = True
        ys: list[int] = []
        xs: list[int] = []

        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))

        regions.append((np.asarray(ys, dtype=np.int32), np.asarray(xs, dtype=np.int32)))

    return regions


def thresholds_from_maps(heat_maps: list[np.ndarray], num_thresholds: int) -> np.ndarray:
    scores = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in heat_maps])
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.array([0.0], dtype=np.float64)

    lo = float(scores.min())
    hi = float(scores.max())
    eps = max(1e-12, abs(hi - lo) * 1e-6, abs(hi) * 1e-9)
    if hi <= lo:
        return np.array([hi + eps, lo - eps], dtype=np.float64)

    n = max(2, int(num_thresholds))
    quantiles = np.linspace(0.0, 1.0, n)
    thresholds = np.quantile(scores, 1.0 - quantiles)
    thresholds = np.concatenate(([hi + eps], thresholds, [lo - eps]))
    thresholds = np.unique(thresholds)
    return thresholds[::-1].astype(np.float64)


def clip_curve_at_fpr(curve: list[dict], fpr_limit: float) -> list[dict]:
    if not curve:
        return []

    pairs = sorted(
        (float(r["fpr"]), float(r["pro"]))
        for r in curve
        if math.isfinite(float(r["fpr"])) and math.isfinite(float(r["pro"]))
    )
    if not pairs:
        return []

    dedup_fpr: list[float] = []
    dedup_pro: list[float] = []
    for fpr, pro in pairs:
        if dedup_fpr and abs(fpr - dedup_fpr[-1]) <= 1e-12:
            dedup_pro[-1] = max(dedup_pro[-1], pro)
        else:
            dedup_fpr.append(fpr)
            dedup_pro.append(pro)

    fpr_arr = np.asarray(dedup_fpr, dtype=np.float64)
    pro_arr = np.maximum.accumulate(np.asarray(dedup_pro, dtype=np.float64))
    limit = float(fpr_limit)

    if fpr_arr[0] > 0.0:
        fpr_arr = np.concatenate(([0.0], fpr_arr))
        pro_arr = np.concatenate(([0.0], pro_arr))

    if fpr_arr[-1] < limit:
        fpr_arr = np.concatenate((fpr_arr, [limit]))
        pro_arr = np.concatenate((pro_arr, [pro_arr[-1]]))
    elif not np.any(np.isclose(fpr_arr, limit, rtol=0.0, atol=1e-12)):
        pro_at_limit = float(np.interp(limit, fpr_arr, pro_arr))
        keep = fpr_arr < limit
        fpr_arr = np.concatenate((fpr_arr[keep], [limit]))
        pro_arr = np.concatenate((pro_arr[keep], [pro_at_limit]))
    else:
        keep = fpr_arr <= limit
        fpr_arr = fpr_arr[keep]
        pro_arr = pro_arr[keep]

    return [{"fpr": float(fpr), "pro": float(pro)} for fpr, pro in zip(fpr_arr, pro_arr)]


def aupro_at_fpr(
    heat_maps: list[np.ndarray],
    mask_maps: list[np.ndarray],
    fpr_limit: float,
    num_thresholds: int,
    connectivity: int,
) -> tuple[dict, list[dict], list[dict]]:
    if len(heat_maps) != len(mask_maps):
        raise ValueError("heat_maps and mask_maps must have the same length")
    if not heat_maps:
        empty = {
            "aupro": float("nan"),
            "n_regions": 0,
            "n_normal_pixels": 0,
            "fpr_limit": float(fpr_limit),
            "num_thresholds": int(num_thresholds),
        }
        return empty, [], []

    masks = [np.asarray(x, dtype=bool) for x in mask_maps]
    heats = [np.asarray(x, dtype=np.float32) for x in heat_maps]
    regions_by_image = [connected_regions(mask, int(connectivity)) for mask in masks]
    n_regions = int(sum(len(regions) for regions in regions_by_image))
    normal_pixels = int(sum((~mask).sum() for mask in masks))

    if n_regions == 0 or normal_pixels == 0:
        info = {
            "aupro": float("nan"),
            "n_regions": n_regions,
            "n_normal_pixels": normal_pixels,
            "fpr_limit": float(fpr_limit),
            "num_thresholds": int(num_thresholds),
        }
        return info, [], []

    raw_curve: list[dict] = []
    for threshold in thresholds_from_maps(heats, int(num_thresholds)):
        false_positive_pixels = 0
        pro_total = 0.0

        for heat, mask, regions in zip(heats, masks, regions_by_image):
            pred = heat >= threshold
            false_positive_pixels += int(np.logical_and(pred, ~mask).sum())
            for ys, xs in regions:
                pro_total += float(pred[ys, xs].mean())

        fpr = false_positive_pixels / max(1, normal_pixels)
        pro = pro_total / max(1, n_regions)
        raw_curve.append({"threshold": float(threshold), "fpr": float(fpr), "pro": float(pro)})

    clipped_curve = clip_curve_at_fpr(raw_curve, float(fpr_limit))
    if clipped_curve:
        fpr = np.asarray([r["fpr"] for r in clipped_curve], dtype=np.float64)
        pro = np.asarray([r["pro"] for r in clipped_curve], dtype=np.float64)
        aupro = float(np.trapz(pro, fpr) / max(1e-12, float(fpr_limit)))
    else:
        aupro = float("nan")

    info = {
        "aupro": aupro,
        "n_regions": n_regions,
        "n_normal_pixels": normal_pixels,
        "fpr_limit": float(fpr_limit),
        "num_thresholds": int(num_thresholds),
    }
    return info, raw_curve, clipped_curve
