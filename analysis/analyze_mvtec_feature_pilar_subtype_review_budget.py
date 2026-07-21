#!/usr/bin/env python3
"""Subtype-level MVTec AD review-budget analysis for Feature-PILAR-FCN.

This script uses only saved final image-level scores. It does not train models
or regenerate predictions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CATEGORIES = [
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

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "results" / "reference" / "mvtec" / "feature_wide64"
SCORE_DIR = REFERENCE_DIR / "details"
SUMMARY_FILE = REFERENCE_DIR / "summary_arch_sweep.csv"
OUT_DIR = ROOT / "results" / "generated" / "mvtec"
FIG_DIR = ROOT / "results" / "generated" / "figures"

CSV_OUT = OUT_DIR / "mvtec_feature_pilar_subtype_review_budget.csv"
JSON_OUT = OUT_DIR / "mvtec_feature_pilar_subtype_review_budget.json"
FIG_PDF = FIG_DIR / "mvtec_feature_pilar_subtype_review_budget_rf90.pdf"
FIG_SVG = FIG_DIR / "mvtec_feature_pilar_subtype_review_budget_rf90.svg"


def read_final_scores(category: str) -> pd.DataFrame:
    path = SCORE_DIR / f"results_{category}_layer2+layer3_hard0p1x1_zscore.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    expected = {
        "category",
        "image",
        "defect_type",
        "label",
        "score",
        "feature_layers",
        "hard_loss",
        "score_norm",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if set(df["category"].unique()) != {category}:
        raise ValueError(f"{path} contains unexpected categories: {df['category'].unique()}")
    if set(df["feature_layers"].unique()) != {"layer2,layer3"}:
        raise ValueError(f"{path} does not match final feature-layer policy")
    if set(df["hard_loss"].unique()) != {"0.1:1"}:
        raise ValueError(f"{path} does not match final hard-loss policy")
    if set(df["score_norm"].unique()) != {"zscore"}:
        raise ValueError(f"{path} does not match final score-normalization policy")
    return df


def roc_auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(labels) + 1)
    precision = tp / ranks
    return float(precision[sorted_labels == 1].sum() / n_pos)


def recall_at_fraction(labels: np.ndarray, fraction: float) -> float:
    n = len(labels)
    n_anom = int(labels.sum())
    if n_anom == 0:
        return float("nan")
    k = int(math.ceil(fraction * n))
    if k <= 0:
        return 0.0
    return float(labels[:k].sum() / n_anom)


def review_fraction_at_recall(labels: np.ndarray, target: float) -> float:
    n = len(labels)
    n_anom = int(labels.sum())
    if n_anom == 0:
        return float("nan")
    required = required_anomaly_count(n_anom, target)
    cumulative = np.cumsum(labels)
    idx = np.where(cumulative >= required)[0]
    if len(idx) == 0:
        return 1.0
    return float((idx[0] + 1) / n)


def required_anomaly_count(n_anom: int, target: float) -> int:
    return int(math.ceil(target * n_anom))


def oracle_min_review_fraction(n_anom: int, n_total: int, target: float) -> float:
    if n_anom <= 0 or n_total <= 0:
        return float("nan")
    return float(required_anomaly_count(n_anom, target) / n_total)


def random_expected_review_fraction(n_anom: int, n_total: int, target: float) -> float:
    if n_anom <= 0 or n_total <= 0:
        return float("nan")
    k = required_anomaly_count(n_anom, target)
    return float(k * (n_total + 1) / (n_total * (n_anom + 1)))


def normalized_review_efficiency(observed: float, oracle_min: float, random_expected: float) -> float:
    denominator = random_expected - oracle_min
    if (
        not np.isfinite(observed)
        or not np.isfinite(oracle_min)
        or not np.isfinite(random_expected)
        or denominator <= 0
    ):
        return float("nan")
    return float((random_expected - observed) / denominator)


def compute_subtype_metrics(category: str, df: pd.DataFrame) -> list[dict]:
    normal = df[df["label"].astype(int) == 0].copy()
    anomalous = df[df["label"].astype(int) == 1].copy()
    rows: list[dict] = []
    for subtype, subtype_df in anomalous.groupby("defect_type", sort=True):
        eval_df = pd.concat([normal, subtype_df], ignore_index=True)
        eval_df = eval_df.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
        labels = eval_df["label"].astype(int).to_numpy()
        scores = eval_df["score"].astype(float).to_numpy()
        fractions = np.linspace(0.0, 1.0, 101)
        curve = [
            {
                "review_fraction": float(frac),
                "recall": recall_at_fraction(labels, float(frac)),
            }
            for frac in fractions
        ]
        n_normal = int(len(normal))
        n_anom = int(len(subtype_df))
        n_total = int(n_normal + n_anom)
        rf90 = review_fraction_at_recall(labels, 0.90)
        rf95 = review_fraction_at_recall(labels, 0.95)
        oracle90 = oracle_min_review_fraction(n_anom, n_total, 0.90)
        oracle95 = oracle_min_review_fraction(n_anom, n_total, 0.95)
        random90 = random_expected_review_fraction(n_anom, n_total, 0.90)
        random95 = random_expected_review_fraction(n_anom, n_total, 0.95)
        rows.append(
            {
                "category": category,
                "anomaly_subtype": subtype,
                "n_normal": n_normal,
                "n_anomalies": n_anom,
                "n_total": n_total,
                "image_auroc": roc_auc_from_scores(labels, scores),
                "image_ap": average_precision_from_scores(labels, scores),
                "recall_at_5_percent": recall_at_fraction(labels, 0.05),
                "recall_at_10_percent": recall_at_fraction(labels, 0.10),
                "review_fraction_at_90_percent_recall": rf90,
                "review_fraction_at_95_percent_recall": rf95,
                "oracle_min_review_fraction_at_90_percent_recall": oracle90,
                "oracle_min_review_fraction_at_95_percent_recall": oracle95,
                "random_expected_review_fraction_at_90_percent_recall": random90,
                "random_expected_review_fraction_at_95_percent_recall": random95,
                "normalized_review_efficiency_at_90_percent_recall": normalized_review_efficiency(
                    rf90, oracle90, random90
                ),
                "normalized_review_efficiency_at_95_percent_recall": normalized_review_efficiency(
                    rf95, oracle95, random95
                ),
                "small_sample_flag": bool(n_anom <= 10),
                "curve": curve,
            }
        )
    return rows


def validate_category_level() -> dict:
    expected = {
        "recall_at_5_percent": 0.0771,
        "recall_at_10_percent": 0.1480,
        "review_fraction_at_90_percent_recall": 0.6743,
        "review_fraction_at_95_percent_recall": 0.7255,
    }
    per_category = []
    for category in CATEGORIES:
        df = read_final_scores(category)
        ranked = df.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
        labels = ranked["label"].astype(int).to_numpy()
        per_category.append(
            {
                "category": category,
                "recall_at_5_percent": recall_at_fraction(labels, 0.05),
                "recall_at_10_percent": recall_at_fraction(labels, 0.10),
                "review_fraction_at_90_percent_recall": review_fraction_at_recall(labels, 0.90),
                "review_fraction_at_95_percent_recall": review_fraction_at_recall(labels, 0.95),
            }
        )
    macro = {
        key: float(np.mean([row[key] for row in per_category]))
        for key in expected
    }
    for key, target in expected.items():
        if abs(macro[key] - target) > 5e-4:
            raise ValueError(f"Category-level validation failed for {key}: {macro[key]} vs {target}")
    summary = pd.read_csv(SUMMARY_FILE)
    macro_row = summary[summary["category"] == "macro_mean"].iloc[0]
    if macro_row["architecture"] != "wide64":
        raise ValueError("Final summary does not point to wide64 architecture")
    if macro_row["score_norm"] != "zscore" or float(macro_row["image_topk_frac"]) != 0.01:
        raise ValueError("Final summary does not match z-score/top-1% policy")
    return {"per_category": per_category, "macro": macro}


def write_outputs(rows: list[dict], category_validation: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    csv_rows = [{k: v for k, v in row.items() if k != "curve"} for row in rows]
    efficiency_values = [
        row["normalized_review_efficiency_at_90_percent_recall"]
        for row in csv_rows
    ] + [
        row["normalized_review_efficiency_at_95_percent_recall"]
        for row in csv_rows
    ]
    nonfinite_efficiencies = [
        row
        for row in csv_rows
        if not np.isfinite(row["normalized_review_efficiency_at_90_percent_recall"])
        or not np.isfinite(row["normalized_review_efficiency_at_95_percent_recall"])
    ]
    out_of_expected_range = [
        {
            "category": row["category"],
            "anomaly_subtype": row["anomaly_subtype"],
            "normalized_review_efficiency_at_90_percent_recall": row[
                "normalized_review_efficiency_at_90_percent_recall"
            ],
            "normalized_review_efficiency_at_95_percent_recall": row[
                "normalized_review_efficiency_at_95_percent_recall"
            ],
        }
        for row in csv_rows
        if row["normalized_review_efficiency_at_90_percent_recall"] < -1e-12
        or row["normalized_review_efficiency_at_90_percent_recall"] > 1.0 + 1e-12
        or row["normalized_review_efficiency_at_95_percent_recall"] < -1e-12
        or row["normalized_review_efficiency_at_95_percent_recall"] > 1.0 + 1e-12
    ]
    if nonfinite_efficiencies:
        raise ValueError(f"Non-finite normalized efficiency values: {nonfinite_efficiencies}")
    pd.DataFrame(csv_rows).sort_values(
        [
            "normalized_review_efficiency_at_90_percent_recall",
            "review_fraction_at_90_percent_recall",
            "n_anomalies",
            "category",
            "anomaly_subtype",
        ],
        ascending=[False, True, False, True, True],
    ).to_csv(CSV_OUT, index=False)

    payload = {
        "analysis_name": "mvtec_feature_pilar_subtype_review_budget",
        "source_score_dir": str(SCORE_DIR.relative_to(ROOT)),
        "score_configuration": {
            "model": "Feature-PILAR-FCN",
            "feature_layers": "layer2,layer3",
            "score_norm": "zscore",
            "image_score": "top-1% anomaly map mean",
            "hard_loss": "0.1:1",
            "architecture": "final MVTec architecture reported in manuscript",
        },
        "small_sample_policy": "small_sample_flag is true when n_anomalies <= 10",
        "oracle_min_review_fraction": {
            "at_90_percent_recall": "k_90 / n_total, where k_90 = ceil(0.90 * n_anomalies)",
            "at_95_percent_recall": "k_95 / n_total, where k_95 = ceil(0.95 * n_anomalies)",
        },
        "random_expected_review_fraction": {
            "at_90_percent_recall": "k_90 * (n_total + 1) / (n_total * (n_anomalies + 1))",
            "at_95_percent_recall": "k_95 * (n_total + 1) / (n_total * (n_anomalies + 1))",
        },
        "normalized_review_efficiency": {
            "at_90_percent_recall": "(random_expected_review_fraction_at_90_percent_recall - observed_review_fraction_at_90_percent_recall) / (random_expected_review_fraction_at_90_percent_recall - oracle_min_review_fraction_at_90_percent_recall)",
            "at_95_percent_recall": "(random_expected_review_fraction_at_95_percent_recall - observed_review_fraction_at_95_percent_recall) / (random_expected_review_fraction_at_95_percent_recall - oracle_min_review_fraction_at_95_percent_recall)",
            "interpretation": "1.0 indicates the oracle minimum review fraction, 0.0 corresponds to the exact finite-sample random-ranking expectation, and negative values indicate worse-than-random high-recall efficiency.",
        },
        "normalized_efficiency_validation": {
            "all_finite": bool(all(np.isfinite(v) for v in efficiency_values)),
            "out_of_expected_range": out_of_expected_range,
        },
        "category_level_validation": category_validation,
        "subtypes": rows,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_ranked_rf90(rows: list[dict]) -> None:
    df = pd.DataFrame([{k: v for k, v in row.items() if k != "curve"} for row in rows])
    df = df.sort_values("review_fraction_at_90_percent_recall", ascending=True).reset_index(drop=True)
    labels = [
        f"{r.category} / {r.anomaly_subtype} (n={int(r.n_anomalies)})"
        + ("*" if bool(r.small_sample_flag) else "")
        for r in df.itertuples(index=False)
    ]
    values = df["review_fraction_at_90_percent_recall"].to_numpy() * 100.0
    colors = np.where(df["small_sample_flag"].to_numpy(), "#d08a2e", "#2f6da8")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    height = max(10.0, 0.18 * len(df) + 2.2)
    fig, ax = plt.subplots(figsize=(7.2, height))
    y = np.arange(len(df))
    ax.barh(y, values, color=colors, edgecolor="none", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Review fraction at 90% recall (%)")
    ax.grid(axis="x", color="#e4e4e4", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(
        0,
        1.01,
        "* small sample: n anomalies <= 10",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color="#6a4b1f",
    )
    fig.tight_layout()
    fig.savefig(FIG_PDF, bbox_inches="tight")
    fig.savefig(FIG_SVG, bbox_inches="tight")


def main() -> None:
    category_validation = validate_category_level()
    rows: list[dict] = []
    for category in CATEGORIES:
        df = read_final_scores(category)
        rows.extend(compute_subtype_metrics(category, df))
    write_outputs(rows, category_validation)
    plot_ranked_rf90(rows)
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["normalized_review_efficiency_at_90_percent_recall"],
            row["review_fraction_at_90_percent_recall"],
            -row["n_anomalies"],
            row["category"],
            row["anomaly_subtype"],
        ),
    )
    ranked_raw = sorted(rows, key=lambda row: row["review_fraction_at_90_percent_recall"])
    print(f"categories={len(CATEGORIES)}")
    print(f"subtypes={len(rows)}")
    print("most favorable normalized efficiency:")
    for row in ranked[:5]:
        print(
            row["category"],
            row["anomaly_subtype"],
            row["n_anomalies"],
            f"eff90={row['normalized_review_efficiency_at_90_percent_recall']:.4f}",
            f"auc={row['image_auroc']:.4f}",
            f"rf90={100 * row['review_fraction_at_90_percent_recall']:.2f}",
            f"oracle90={100 * row['oracle_min_review_fraction_at_90_percent_recall']:.2f}",
            f"random90={100 * row['random_expected_review_fraction_at_90_percent_recall']:.2f}",
        )
    print("least favorable normalized efficiency:")
    for row in ranked[-5:][::-1]:
        print(
            row["category"],
            row["anomaly_subtype"],
            row["n_anomalies"],
            f"eff90={row['normalized_review_efficiency_at_90_percent_recall']:.4f}",
            f"auc={row['image_auroc']:.4f}",
            f"rf90={100 * row['review_fraction_at_90_percent_recall']:.2f}",
            f"oracle90={100 * row['oracle_min_review_fraction_at_90_percent_recall']:.2f}",
            f"random90={100 * row['random_expected_review_fraction_at_90_percent_recall']:.2f}",
        )
    print("lowest raw review fractions:")
    for row in ranked_raw[:5]:
        print(
            row["category"],
            row["anomaly_subtype"],
            row["n_anomalies"],
            f"{100 * row['review_fraction_at_90_percent_recall']:.2f}",
            f"{100 * row['recall_at_10_percent']:.2f}",
        )
    print("highest raw review fractions:")
    for row in ranked_raw[-5:][::-1]:
        print(
            row["category"],
            row["anomaly_subtype"],
            row["n_anomalies"],
            f"{100 * row['review_fraction_at_90_percent_recall']:.2f}",
            f"{100 * row['recall_at_10_percent']:.2f}",
        )
    print(CSV_OUT)
    print(JSON_OUT)
    print(FIG_PDF)
    print(FIG_SVG)


if __name__ == "__main__":
    main()
