#!/usr/bin/env python3
"""Recompute PILAR screening curves and MVTec subtype results from image scores."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.verify_reference_results import MVT, VISA, SEEDS, ranking_metrics, rows, verify_matrix


def tied_blocks(score_rows):
    blocks = defaultdict(lambda: [0, 0])
    for row in score_rows:
        label, score = int(row["label"]), float(row["score"])
        if label not in (0, 1) or not math.isfinite(score):
            raise ValueError("Scores must be finite and labels binary")
        blocks[score][0] += 1
        blocks[score][1] += label
    return [blocks[score] for score in sorted(blocks, reverse=True)]


def review_fraction(score_rows, target):
    """Expected rank of the kth anomaly under uniform order within each tie."""
    if not 0 <= target <= 1:
        raise ValueError("Recall target must lie in [0,1]")
    blocks = tied_blocks(score_rows)
    n, m = sum(b for b, _ in blocks), sum(r for _, r in blocks)
    if not n or not m:
        raise ValueError("Review fractions require at least one anomaly")
    k = math.ceil(target * m)
    if not k:
        return 0.0
    reviewed = found = 0
    for size, positives in blocks:
        if found + positives >= k:
            return (reviewed + (k - found) * (size + 1) / (positives + 1)) / n
        reviewed += size
        found += positives
    raise AssertionError("Unreachable recall target")


def recall_at_budget(score_rows, budget):
    """Expected anomaly recall after ceil(budget*N) reviews, including ties."""
    if not 0 <= budget <= 1:
        raise ValueError("Review budget must lie in [0,1]")
    blocks = tied_blocks(score_rows)
    n, m = sum(b for b, _ in blocks), sum(r for _, r in blocks)
    if not n or not m:
        raise ValueError("Recall requires at least one anomaly")
    remaining, found = math.ceil(budget * n), 0.0
    for size, positives in blocks:
        take = min(remaining, size)
        found += take * positives / size
        remaining -= take
        if not remaining:
            break
    return found / m


def review_metrics(score_rows):
    n = len(score_rows)
    m = sum(int(r["label"]) for r in score_rows)
    auc, ap = ranking_metrics(score_rows)
    values = dict(image_auc=auc, image_ap=ap)
    for target in (0.90, 0.95):
        k = math.ceil(target * m)
        oracle = k / n
        random = k * (n + 1) / (n * (m + 1))
        observed = review_fraction(score_rows, target)
        suffix = str(round(100 * target))
        values.update({f"rf{suffix}": observed, f"oracle{suffix}": oracle, f"random{suffix}": random, f"eta{suffix}": (random - observed) / (random - oracle) if random > oracle else None})
    values.update(recall5=recall_at_budget(score_rows, 0.05), recall10=recall_at_budget(score_rows, 0.10))
    return values


def summarize(records, metrics, identity):
    categories = {tuple(r[k] for k in identity) for r in records}
    expected = {(seed, key) for seed in SEEDS for key in categories}
    if len(records) != len(expected) or {(r["seed"], tuple(r[k] for k in identity)) for r in records} != expected:
        raise ValueError("Cannot aggregate an incomplete seed/category matrix")
    macros = [dict(seed=seed, **{metric: statistics.mean(r[metric] for r in records if r["seed"] == seed) for metric in metrics}) for seed in SEEDS]
    summary = {metric: dict(mean=statistics.mean(r[metric] for r in macros), sample_sd=statistics.stdev(r[metric] for r in macros), n_seeds=5) for metric in metrics}
    return macros, summary


def save_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def analyze(score_rows, output_dir):
    groups = defaultdict(list)
    seen = set()
    for row in score_rows:
        key = row["dataset"], int(row["seed"]), row["category"]
        image_key = (*key, row["image"])
        if image_key in seen:
            raise ValueError(f"Duplicate image: {image_key}")
        seen.add(image_key)
        groups[key].append(row)
    datasets = sorted({key[0] for key in groups})
    if not datasets or set(datasets) - {"mvtec", "visa"}:
        raise ValueError("Expected MVTec and/or VisA score rows")
    summaries = {}
    all_curves = []
    for dataset in datasets:
        categories = MVT if dataset == "mvtec" else VISA
        verify_matrix([dict(seed=s, category=c) for d, s, c in groups if d == dataset], categories, SEEDS, dataset)
        category_rows = []
        subtypes = []
        for seed in SEEDS:
            for category in categories:
                selected = groups[dataset, seed, category]
                values = review_metrics(selected)
                category_rows.append(dict(seed=seed, category=category, **values))
                for budget_percent in range(101):
                    all_curves.append(dict(dataset=dataset, seed=seed, category=category, review_fraction=budget_percent / 100, recall=recall_at_budget(selected, budget_percent / 100)))
                if dataset == "mvtec":
                    defects = sorted({r["defect_type"] for r in selected if int(r["label"]) == 1})
                    for subtype in defects:
                        subset = [r for r in selected if int(r["label"]) == 0 or r["defect_type"] == subtype]
                        subtypes.append(dict(seed=seed, category=category, subtype=subtype, n_normal=sum(int(r["label"]) == 0 for r in subset), n_anomaly=sum(int(r["label"]) == 1 for r in subset), **review_metrics(subset)))
        metrics = tuple(k for k in category_rows[0] if k not in ("seed", "category"))
        macros, summary = summarize(category_rows, metrics, ("category",))
        summaries[dataset] = summary
        save_csv(output_dir / dataset / "review_budget_per_category_seed.csv", category_rows)
        save_csv(output_dir / dataset / "review_budget_seed_macros.csv", macros)
        if subtypes:
            if len(subtypes) != 365 or len({(r['category'],r['subtype']) for r in subtypes}) != 73:
                raise ValueError("Expected 73 MVTec subtypes in each of five seeds")
            subtype_macros, subtype_summary = summarize(subtypes, metrics, ("category", "subtype"))
            summaries["mvtec_subtypes"] = subtype_summary
            save_csv(output_dir / dataset / "subtypes.csv", subtypes)
            save_csv(output_dir / dataset / "subtype_seed_macros.csv", subtype_macros)
    save_csv(output_dir / "review_curves_per_category_seed.csv", all_curves)
    curve_statistics = []
    for dataset in datasets:
        for budget_percent in range(101):
            selected = [r for r in all_curves if r["dataset"] == dataset and r["review_fraction"] == budget_percent / 100]
            seed_means = [statistics.mean(r["recall"] for r in selected if r["seed"] == seed) for seed in SEEDS]
            curve_statistics.append(dict(dataset=dataset, review_fraction=budget_percent / 100, recall_mean=statistics.mean(seed_means), recall_sample_sd=statistics.stdev(seed_means)))
    save_csv(output_dir / "review_curves.csv", curve_statistics)
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "results/reference/paper/image_scores.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/generated/review_budget")
    args = parser.parse_args()
    summaries = analyze(rows(args.scores), args.output_dir)
    print(f"Wrote screening results to {args.output_dir}")
    for key, values in summaries.items():
        print(f"{key}: review fraction at 90% recall = {values['rf90']['mean']:.6f} ± {values['rf90']['sample_sd']:.6f} (five seeds)")


if __name__ == "__main__":
    main()
