#!/usr/bin/env python3
"""Verify complete paper matrices and seed-level aggregation without training."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "reference"
PAPER = REFERENCE / "paper"
SEEDS = tuple(range(42, 47))
MVT = tuple("bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper".split())
VISA = tuple("candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum".split())
METRICS = ("image_auc", "image_ap", "pixel_auc", "aupro_30")
STRICT_VARIANTS = {"baseline_recon_only", "inhib_none", "inhib_radial_all", "enc_channels_1", "enc_channels_2", "enc_channels_4", "dec_channels_4"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(label: str, actual: float, expected: float, tolerance: float = 5e-12) -> None:
    if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > tolerance:
        raise AssertionError(f"{label}: expected {expected}, found {actual}")


def verify_matrix(records, categories, seeds, label):
    expected = {(int(s), str(c)) for s in seeds for c in categories}
    seen = [(int(r["seed"]), str(r["category"])) for r in records]
    if set(seen) != expected or len(seen) != len(expected):
        raise AssertionError(f"{label}: incomplete or duplicate category/seed matrix")


def verify_summaries(records, summary_path, metrics):
    summaries = json.loads(summary_path.read_text())
    keys = {(r["dataset"], r["variant"]) for r in records}
    if {(r["dataset"], r["variant"]) for r in summaries} != keys or len(summaries) != len(keys):
        raise AssertionError(f"Unexpected groups in {summary_path.name}")
    for summary in summaries:
        group = [r for r in records if (r["dataset"], r["variant"]) == (summary["dataset"], summary["variant"])]
        label = f"{summary['dataset']}/{summary['variant']}"
        verify_matrix(group, summary["categories"], summary["seeds"], label)
        for metric in metrics:
            if not all(r.get(metric, "") != "" for r in group):
                if metric in summary["metrics"]:
                    raise AssertionError(f"Unexpected {metric} for {label}")
                continue
            values = [float(r[metric]) for r in group]
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
                raise AssertionError(f"Invalid {metric} for {label}")
            macros = {s: statistics.mean(float(r[metric]) for r in group if int(r["seed"]) == s) for s in summary["seeds"]}
            aggregate = summary["metrics"][metric]
            close(f"{label}/{metric}/mean", aggregate["mean"], statistics.mean(macros.values()))
            if aggregate["n_seeds"] != len(macros):
                raise AssertionError(f"Wrong seed count in {label}")
            if len(macros) == 1:
                if aggregate["sample_sd"] is not None:
                    raise AssertionError("Single-seed comparisons must not imply seed uncertainty")
            else:
                close(f"{label}/{metric}/SD", aggregate["sample_sd"], statistics.stdev(macros.values()))
            if len(summary["seed_macros"]) != len(macros):
                raise AssertionError(f"Wrong number of seed macros in {label}")
            for row in summary["seed_macros"]:
                close(f"{label}/{metric}/seed{row['seed']}", row[metric], macros[row["seed"]])
    return summaries


def ranking_metrics(score_rows):
    """Independent grouped-score reference; anomaly is positive, high is anomalous."""
    blocks = defaultdict(lambda: [0, 0])
    for row in score_rows:
        score, label = float(row["score"]), int(row["label"])
        if not math.isfinite(score) or label not in (0, 1):
            raise AssertionError("Invalid image score")
        blocks[score][label] += 1
    n_positive = sum(n[1] for n in blocks.values())
    n_negative = sum(n[0] for n in blocks.values())
    if not n_positive or not n_negative:
        raise AssertionError("Both image labels are required")
    wins = negatives_before = 0.0
    for score in sorted(blocks):
        negative, positive = blocks[score]
        wins += positive * (negatives_before + negative / 2)
        negatives_before += negative
    ap = cumulative_positive = reviewed = 0.0
    for score in sorted(blocks, reverse=True):
        negative, positive = blocks[score]
        cumulative_positive += positive
        reviewed += negative + positive
        ap += positive / n_positive * cumulative_positive / reviewed
    return wins / (n_positive * n_negative), ap


def verify() -> list[str]:
    checked = []
    industrial = rows(PAPER / "industrial_category_metrics.csv")
    expected_groups = {("mvtec", v) for v in ("main", "no_inhibition", "no_hard_loss", "layer2_only", "layer3_only")} | {("visa", "main")}
    if {(r["dataset"], r["variant"]) for r in industrial} != expected_groups:
        raise AssertionError("Unexpected industrial experiment")
    for dataset, variant in expected_groups:
        group = [r for r in industrial if (r["dataset"], r["variant"]) == (dataset, variant)]
        verify_matrix(group, MVT if dataset == "mvtec" else VISA, SEEDS if variant == "main" else (42,), f"{dataset}/{variant}")
        if any(int(r["epochs"]) != 10 or int(r["image_size"]) != 512 for r in group):
            raise AssertionError("Industrial references must use the paper's ten-epoch, 512-pixel protocol")
    verify_summaries(industrial, PAPER / "industrial_summary.json", METRICS)
    checked.append("MVTec 75, VisA 60, and MVTec ablation 60 category/seed results")
    groups = defaultdict(list)
    score_rows = rows(PAPER / "image_scores.csv")
    identities = set()
    for r in score_rows:
        identity = (r["dataset"], int(r["seed"]), r["category"], r["image"])
        if identity in identities or Path(r["image"]).is_absolute():
            raise AssertionError("Duplicate or non-portable image identity")
        identities.add(identity)
        groups[identity[:3]].append(r)
    main = [r for r in industrial if r["variant"] == "main"]
    if set(groups) != {(r["dataset"], int(r["seed"]), r["category"]) for r in main}:
        raise AssertionError("Scores do not cover all main category/seed results")
    for dataset, count in (("mvtec", 1725), ("visa", 2162)):
        baseline = None
        for seed in SEEDS:
            selected = {(r["category"], r["image"], r["label"]) for r in score_rows if r["dataset"] == dataset and int(r["seed"]) == seed}
            if len(selected) != count or (baseline is not None and selected != baseline):
                raise AssertionError(f"Inconsistent official test set for {dataset}, seed {seed}")
            baseline = selected
    for r in main:
        auc, ap = ranking_metrics(groups[r["dataset"], int(r["seed"]), r["category"]])
        close("image AUROC from scores", auc, float(r["image_auc"]))
        close("image AP from scores", ap, float(r["image_ap"]))
    checked.append("19,435 image scores: complete identities and independently recalculated AUROC/AP")
    supplementary = rows(PAPER / "supplementary_class_metrics.csv")
    expected_supp = {("mnist", "strict"), ("fashion_mnist", "strict"), ("cifar10", "strict"), ("fashion_mnist", "feature"), ("cifar10", "feature")}
    if {(r["dataset"], r["variant"]) for r in supplementary} != expected_supp:
        raise AssertionError("Unexpected supplementary experiment")
    for key in expected_supp:
        verify_matrix([r for r in supplementary if (r["dataset"], r["variant"]) == key], range(10), SEEDS, str(key))
    verify_summaries(supplementary, PAPER / "supplementary_summary.json", METRICS[:2])
    checked.append("Five supplementary experiments: category macro, then sample SD over five seeds")
    ablation = rows(PAPER / "strict_ablation_class_metrics.csv")
    if {(r["dataset"], r["variant"]) for r in ablation} != {(d, v) for d in ("mnist", "fashion_mnist") for v in STRICT_VARIANTS}:
        raise AssertionError("Unexpected strict-image ablation")
    for dataset in ("mnist", "fashion_mnist"):
        for variant in STRICT_VARIANTS:
            verify_matrix([r for r in ablation if (r["dataset"], r["variant"]) == (dataset, variant)], range(10), SEEDS, f"{dataset}/{variant}")
    verify_summaries(ablation, PAPER / "strict_ablation_summary.json", ("image_auc",))
    checked.append("700 strict-image component-ablation class results")
    for regime in ("whole_rgb", "patch_rgb"):
        data = rows(REFERENCE / "mvtec" / regime / "summary_aupro.csv")
        classes = [r for r in data if r["category"] != "macro_mean"]
        if len(classes) != 15 or {r["category"] for r in classes} != set(MVT):
            raise AssertionError(f"Incomplete {regime}")
        macro = [r for r in data if r["category"] == "macro_mean"]
        if len(macro) != 1:
            raise AssertionError(f"Missing macro for {regime}")
        for metric in METRICS:
            close(f"{regime}/{metric}", float(macro[0][metric]), statistics.mean(float(r[metric]) for r in classes))
    checked.append("Whole-image and patchwise RGB reconstruction-domain controls")
    subtypes = rows(PAPER / "mvtec/pilar_subtypes.csv")
    subtype_keys = {(r["category"], r["subtype"]) for r in subtypes}
    if len(subtype_keys) != 73 or len(subtypes) != 365 or {(int(r['seed']), r['category'], r['subtype']) for r in subtypes} != {(s, c, t) for s in SEEDS for c, t in subtype_keys}:
        raise AssertionError("Incomplete 73-subtype × five-seed reference")
    for r in subtypes:
        n = int(r["n_normal"]) + int(r["n_anomaly"])
        m = int(r["n_anomaly"])
        for percentage in (90, 95):
            k = math.ceil(percentage / 100 * m)
            close("subtype oracle", float(r[f"oracle{percentage}"]), k / n)
            close("subtype random", float(r[f"random{percentage}"]), k * (n + 1) / (n * (m + 1)))
    checked.append("365 subtype results and finite-population review-budget formulas")
    return checked


if __name__ == "__main__":
    checked = verify()
    print(f"Reference verification passed ({len(checked)} groups):")
    for label in checked:
        print(f"- {label}")
