#!/usr/bin/env python3
"""Verify the compact reference inventory and manuscript-level aggregates."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "reference"


def _rows(relative: str) -> list[dict[str, str]]:
    path = REFERENCE / relative
    if not path.is_file():
        raise AssertionError(f"Missing reference file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row(relative: str, key: str, value: str) -> dict[str, str]:
    matches = [row for row in _rows(relative) if row.get(key) == value]
    if len(matches) != 1:
        raise AssertionError(f"Expected one {key}={value!r} row in {relative}, found {len(matches)}")
    return matches[0]


def _close(label: str, actual: str | float, expected: float, tolerance: float = 5e-10) -> None:
    value = float(actual)
    if abs(value - expected) > tolerance:
        raise AssertionError(f"{label}: expected {expected}, found {value}")


def verify() -> list[str]:
    checked: list[str] = []

    mvtec = _row("mvtec/feature_wide64/summary_arch_sweep.csv", "category", "macro_mean")
    for key, expected in {
        "image_auc": 0.9694874561845537,
        "image_ap": 0.9887466424941189,
        "pixel_auc": 0.9653976273119519,
        "aupro_30": 0.9264611984953862,
        "backbone_params_frozen": 2782784.0,
        "lipnet_params": 1547782.0,
        "approx_infer_flops_per_image": 16085945856.0,
    }.items():
        _close(f"MVTec Feature-PILAR {key}", mvtec[key], expected)
    detail_files = sorted((REFERENCE / "mvtec/feature_wide64/details").glob("results_*.csv"))
    if len(detail_files) != 15:
        raise AssertionError(f"Expected 15 final MVTec detail files, found {len(detail_files)}")
    checked.append("MVTec Feature-PILAR-FCN")

    for regime, expected in (("whole_rgb", 0.6958893086780493), ("patch_rgb", 0.6965822351960423)):
        row = _row(f"mvtec/{regime}/summary_aupro.csv", "category", "macro_mean")
        _close(f"MVTec {regime} image_auc", row["image_auc"], expected)
        checked.append(f"MVTec {regime}")

    for dataset, expected_auc, expected_ap in (
        ("cifar10", 0.8124055355555556, 0.9696587985830435),
        ("fashion_mnist", 0.9361119355555556, 0.9915748345314372),
    ):
        relative = f"{dataset}/feature/summary.csv"
        _close(f"{dataset} feature AUC", _row(relative, "metric", "auc")["mean"], expected_auc)
        _close(f"{dataset} feature AP", _row(relative, "metric", "ap")["mean"], expected_ap)
        checked.append(f"{dataset} Feature-PILAR-FCN")

    grayscale = "strict/mnist_fashion_mnist/summary_method_metrics.csv"
    _close("Fashion-MNIST strict AUC", _row(grayscale, "dataset", "fashion_mnist")["mean_auc_combo"], 0.8759886133333333)
    _close("MNIST strict AUC", _row(grayscale, "dataset", "mnist")["mean_auc_combo"], 0.9464600994620425)
    cifar = _row("strict/cifar10/summary_method_metrics.csv", "dataset", "cifar10")
    _close("CIFAR-10 strict AUC", cifar["mean_auc_combo"], 0.5735264266666666)
    checked.append("strict image regimes")

    subtypes = _rows("mvtec/feature_wide64/mvtec_feature_pilar_subtype_review_budget.csv")
    if len(subtypes) != 73:
        raise AssertionError(f"Expected 73 MVTec defect subtypes, found {len(subtypes)}")
    oracle_count = sum(
        abs(float(row["normalized_review_efficiency_at_90_percent_recall"]) - 1.0) <= 1e-12
        for row in subtypes
    )
    if oracle_count != 37:
        raise AssertionError(f"Expected 37 oracle-efficient subtypes at 90% recall, found {oracle_count}")
    checked.append("MVTec subtype review budget")

    feature_ablation = _rows("mvtec/feature_ablation/summary.csv")
    if len(feature_ablation) != 36:
        raise AssertionError(f"Expected 36 MVTec feature-ablation rows, found {len(feature_ablation)}")
    if {row["category"] for row in feature_ablation} != {"screw", "carpet"}:
        raise AssertionError("MVTec feature ablation must contain only screw and carpet")
    checked.append("MVTec feature-domain ablation")

    if not (REFERENCE / "ablation/s4_noproto_ablation_summary.csv").is_file():
        raise AssertionError("Missing strict component-ablation summary")
    checked.append("strict component ablation")
    return checked


def main() -> None:
    checked = verify()
    print(f"Reference verification passed ({len(checked)} groups):")
    for label in checked:
        print(f"- {label}")


if __name__ == "__main__":
    main()
