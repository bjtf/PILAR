#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np

from lipnet.metrics import roc_auc


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


SCORE_RE = re.compile(r"scores_seed(?P<seed>\d+)_class(?P<class_id>\d+)_(?P<norm>[^.]+)\.csv$")


def _read_scores(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)


def _matrix_rows(values: dict[tuple[int, int], float], diagonal: float, value_name: str) -> list[dict]:
    rows = []
    for normal_id, normal_name in enumerate(CIFAR10_CLASSES):
        row = {"normal_class": normal_id, "normal_name": normal_name}
        for anomaly_id, anomaly_name in enumerate(CIFAR10_CLASSES):
            if anomaly_id == normal_id:
                val = diagonal
            else:
                val = values.get((normal_id, anomaly_id), float("nan"))
            row[f"{anomaly_id}_{anomaly_name}"] = val
        rows.append(row)
    return rows


def _plot_heatmap(matrix: np.ndarray, title: str, output: Path, cmap: str, vmin: float, vmax: float) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=160)
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Anomaly class")
        ax.set_ylabel("Normal training class")
        ax.set_xticks(range(len(CIFAR10_CLASSES)), CIFAR10_CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(len(CIFAR10_CLASSES)), CIFAR10_CLASSES)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                txt = "-" if i == j else f"{matrix[i, j]:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        plt.close(fig)
        return
    except ModuleNotFoundError:
        pass

    from PIL import Image, ImageDraw, ImageFont

    stops = (
        [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]
        if cmap == "viridis"
        else [(0, 0, 4), (74, 16, 114), (189, 55, 84), (251, 136, 97), (252, 253, 191)]
    )

    def color_for(value: float) -> tuple[int, int, int]:
        t = (float(value) - float(vmin)) / max(1e-12, float(vmax) - float(vmin))
        t = min(1.0, max(0.0, t))
        pos = t * (len(stops) - 1)
        i = min(len(stops) - 2, int(pos))
        frac = pos - i
        a, b = stops[i], stops[i + 1]
        return tuple(int(round(a[k] + frac * (b[k] - a[k]))) for k in range(3))

    n = len(CIFAR10_CLASSES)
    cell = 74
    left = 145
    top = 95
    right_pad = 30
    bottom_pad = 120
    width = left + n * cell + right_pad
    height = top + n * cell + bottom_pad
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 13)
        small = ImageFont.truetype("Arial.ttf", 11)
        title_font = ImageFont.truetype("Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
        title_font = ImageFont.load_default()

    draw.text((left, 20), title, fill="black", font=title_font)
    draw.text((left + n * cell // 2 - 45, top + n * cell + 72), "Anomaly class", fill="black", font=font)
    draw.text((12, top + n * cell // 2 - 12), "Normal class", fill="black", font=font)

    for j, name in enumerate(CIFAR10_CLASSES):
        x = left + j * cell + cell // 2
        draw.text((x - 20, top - 40), name[:8], fill="black", font=small)
    for i, name in enumerate(CIFAR10_CLASSES):
        y = top + i * cell + cell // 2 - 7
        draw.text((left - 100, y), name[:10], fill="black", font=small)

    for i in range(n):
        for j in range(n):
            x0 = left + j * cell
            y0 = top + i * cell
            val = float(matrix[i, j])
            color = (232, 232, 232) if i == j else color_for(val)
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline=(220, 220, 220))
            txt = "-" if i == j else f"{val:.2f}"
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            txt_color = "black" if brightness > 145 else "white"
            draw.text((x0 + 23, y0 + 29), txt, fill=txt_color, font=small)

    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CIFAR-10 pairwise anomaly confusion matrices from scores.")
    parser.add_argument("--scores_dir", required=True)
    parser.add_argument("--score_norm", default="zscore")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    output_dir = Path(args.output_dir) if args.output_dir else scores_dir
    norm = str(args.score_norm)

    long_rows: list[dict] = []
    per_pair: dict[tuple[int, int], list[float]] = defaultdict(list)
    files = sorted(scores_dir.glob(f"scores_seed*_class*_{norm}.csv"))
    if not files:
        raise FileNotFoundError(f"No score files matching scores_seed*_class*_{norm}.csv in {scores_dir}")

    for path in files:
        match = SCORE_RE.match(path.name)
        if not match:
            continue
        seed = int(match.group("seed"))
        normal_id = int(match.group("class_id"))
        rows = _read_scores(path)
        normal_scores = np.array([float(r["score"]) for r in rows if int(r["true_class"]) == normal_id], dtype=np.float64)
        if normal_scores.size == 0:
            continue
        for anomaly_id in range(len(CIFAR10_CLASSES)):
            if anomaly_id == normal_id:
                continue
            anomaly_scores = np.array([float(r["score"]) for r in rows if int(r["true_class"]) == anomaly_id], dtype=np.float64)
            if anomaly_scores.size == 0:
                continue
            scores = np.concatenate([normal_scores, anomaly_scores])
            labels = np.concatenate(
                [
                    np.zeros(normal_scores.shape[0], dtype=np.int64),
                    np.ones(anomaly_scores.shape[0], dtype=np.int64),
                ]
            )
            auc = float(roc_auc(scores, labels))
            confusion = float(1.0 - auc)
            per_pair[(normal_id, anomaly_id)].append(auc)
            long_rows.append(
                {
                    "seed": seed,
                    "normal_class": normal_id,
                    "normal_name": CIFAR10_CLASSES[normal_id],
                    "anomaly_class": anomaly_id,
                    "anomaly_name": CIFAR10_CLASSES[anomaly_id],
                    "pairwise_auc": auc,
                    "confusion_1_minus_auc": confusion,
                    "n_normal": int(normal_scores.size),
                    "n_anomaly": int(anomaly_scores.size),
                    "score_norm": norm,
                }
            )

    pair_means = {k: float(np.mean(v)) for k, v in per_pair.items()}
    pair_stds = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in per_pair.items()}
    summary_rows = []
    for normal_id, anomaly_id in sorted(pair_means):
        auc = pair_means[(normal_id, anomaly_id)]
        summary_rows.append(
            {
                "normal_class": normal_id,
                "normal_name": CIFAR10_CLASSES[normal_id],
                "anomaly_class": anomaly_id,
                "anomaly_name": CIFAR10_CLASSES[anomaly_id],
                "pairwise_auc_mean": auc,
                "pairwise_auc_std": pair_stds[(normal_id, anomaly_id)],
                "confusion_1_minus_auc_mean": 1.0 - auc,
                "n": len(per_pair[(normal_id, anomaly_id)]),
                "score_norm": norm,
            }
        )

    class_cols = [f"{i}_{name}" for i, name in enumerate(CIFAR10_CLASSES)]
    _write_csv(
        output_dir / "cifar10_pairwise_auc_long.csv",
        long_rows,
        [
            "seed",
            "normal_class",
            "normal_name",
            "anomaly_class",
            "anomaly_name",
            "pairwise_auc",
            "confusion_1_minus_auc",
            "n_normal",
            "n_anomaly",
            "score_norm",
        ],
    )
    _write_csv(
        output_dir / "cifar10_pairwise_auc_summary.csv",
        summary_rows,
        [
            "normal_class",
            "normal_name",
            "anomaly_class",
            "anomaly_name",
            "pairwise_auc_mean",
            "pairwise_auc_std",
            "confusion_1_minus_auc_mean",
            "n",
            "score_norm",
        ],
    )
    _write_csv(
        output_dir / "cifar10_pairwise_auc_matrix.csv",
        _matrix_rows(pair_means, diagonal=1.0, value_name="pairwise_auc_mean"),
        ["normal_class", "normal_name"] + class_cols,
    )
    confusion_means = {k: 1.0 - v for k, v in pair_means.items()}
    _write_csv(
        output_dir / "cifar10_pairwise_confusion_matrix.csv",
        _matrix_rows(confusion_means, diagonal=0.0, value_name="confusion_1_minus_auc_mean"),
        ["normal_class", "normal_name"] + class_cols,
    )

    mat_auc = np.ones((len(CIFAR10_CLASSES), len(CIFAR10_CLASSES)), dtype=np.float64)
    mat_conf = np.zeros_like(mat_auc)
    for (i, j), auc in pair_means.items():
        mat_auc[i, j] = auc
        mat_conf[i, j] = 1.0 - auc

    if bool(args.plot):
        _plot_heatmap(
            mat_auc,
            f"CIFAR-10 Pairwise AUC ({norm})",
            output_dir / "cifar10_pairwise_auc_heatmap.png",
            cmap="viridis",
            vmin=0.5,
            vmax=1.0,
        )
        _plot_heatmap(
            mat_conf,
            f"CIFAR-10 Pairwise Confusion = 1 - AUC ({norm})",
            output_dir / "cifar10_pairwise_confusion_heatmap.png",
            cmap="magma",
            vmin=0.0,
            vmax=0.5,
        )

    worst = sorted(summary_rows, key=lambda r: float(r["pairwise_auc_mean"]))[:10]
    print(f"Saved pairwise matrices to {output_dir}")
    print("Most confused pairs by pairwise AUC:")
    for row in worst:
        print(
            f"normal={row['normal_name']:<10} anomaly={row['anomaly_name']:<10} "
            f"auc={float(row['pairwise_auc_mean']):.4f} "
            f"conf={float(row['confusion_1_minus_auc_mean']):.4f}"
        )


if __name__ == "__main__":
    main()
