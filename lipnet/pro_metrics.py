from __future__ import annotations

import math

import numpy as np

EVALUATOR_VERSION = "aupro-quantile-v1"


def _validate_connectivity(connectivity: int) -> None:
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")


def _validate_limit(fpr_limit: float) -> float:
    limit = float(fpr_limit)
    if not math.isfinite(limit) or not 0.0 < limit <= 1.0:
        raise ValueError("fpr_limit must be finite and in (0, 1]")
    return limit


def _validate_threshold_count(num_thresholds: int) -> int:
    if isinstance(num_thresholds, (bool, np.bool_)) or int(num_thresholds) != num_thresholds or num_thresholds < 2:
        raise ValueError("num_thresholds must be an integer >= 2")
    return int(num_thresholds)


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    raw = np.asarray(mask)
    if raw.ndim != 2 or not raw.size:
        raise ValueError(f"Expected nonempty 2-D mask, got shape={raw.shape}")
    if not np.isin(raw, [0, 1]).all():
        raise ValueError("Masks must contain only binary values {0, 1}")
    return raw.astype(bool)


def connected_regions(mask: np.ndarray, connectivity: int) -> list[tuple[np.ndarray, np.ndarray]]:
    _validate_connectivity(connectivity)
    mask = _binary_mask(mask)
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
    n = _validate_threshold_count(num_thresholds)
    if not heat_maps:
        raise ValueError("heat_maps must not be empty")
    scores = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in heat_maps])
    if not scores.size or not np.isfinite(scores).all():
        raise ValueError("Heat maps must contain finite scores and must not be empty")

    lo = float(scores.min())
    hi = float(scores.max())
    eps = max(1e-12, abs(hi - lo) * 1e-6, abs(hi) * 1e-9)
    if hi <= lo:
        return np.array([hi + eps, lo - eps], dtype=np.float64)

    quantiles = np.linspace(0.0, 1.0, n)
    thresholds = np.quantile(scores, 1.0 - quantiles)
    thresholds = np.concatenate(([hi + eps], thresholds, [lo - eps]))
    thresholds = np.unique(thresholds)
    return thresholds[::-1].astype(np.float64)


def clip_curve_at_fpr(curve: list[dict], fpr_limit: float) -> list[dict]:
    limit = _validate_limit(fpr_limit)
    if not curve:
        return []

    pairs = sorted((float(r["fpr"]), float(r["pro"])) for r in curve)
    if any(not math.isfinite(x) or not 0.0 <= x <= 1.0 for pair in pairs for x in pair):
        raise ValueError("Curve FPR and PRO values must be finite and in [0, 1]")

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
    """Normalized quantile-sampled PRO area used in the paper.

    Each connected defect region has equal weight. The curve uses 300
    quantile thresholds in the paper, takes the maximum PRO at duplicate
    FPRs, and linearly interpolates at FPR 0.30 before integration. This is
    the paper's evaluator, not an exact all-score-threshold AUPRO integral.
    Undefined cases (no regions or no normal pixels) return NaN explicitly.
    """
    _validate_limit(fpr_limit)
    _validate_threshold_count(num_thresholds)
    _validate_connectivity(connectivity)
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

    masks = [_binary_mask(x) for x in mask_maps]
    with np.errstate(over="ignore", invalid="ignore"):
        heats = [np.asarray(x, dtype=np.float32) for x in heat_maps]
    for heat, mask in zip(heats, masks):
        if heat.shape != mask.shape:
            raise ValueError(f"Heat map and mask shapes differ: {heat.shape} vs {mask.shape}")
        if not np.isfinite(heat).all():
            raise ValueError("Heat maps must be finite in float32")
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
        aupro = float(np.trapezoid(pro, fpr) / float(fpr_limit))
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
