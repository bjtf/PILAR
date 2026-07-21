#!/usr/bin/env python3
import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_DATASET_SPECS = {
    "fashion_mnist": {
        "config": "configs/benchmark_pyramid_fashion.json",
        "run_id": "",
    },
    "mnist": {
        "config": "configs/benchmark_pyramid.json",
        "run_id": "",
    },
    "cifar10": {
        "config": "configs/benchmark_pyramid_cifar10.json",
        "run_id": "",
    },
}


BASE_RECON_ONLY = {
    "decoder_type": "pure_pyramid",
    "channels": "16,8",
    "loss_type": "mse_l1",
    "l1_weight": 0.2,
    "epochs_mult": 2,
    "score_znorm": True,
    "recon_only": True,
    "latent_score_weight": 0.0,
    "combo_sign_calibration": "off",
}


ABLATIONS = [
    {
        "name": "baseline_recon_only",
        "factor": "baseline",
        "description": "S4 backbone with reconstruction-only scoring (no prototypes)",
        "override": {},
        "remove": [],
    },
    {
        "name": "inhib_none",
        "factor": "inhibition",
        "description": "Disable lateral inhibition in all encoder layers",
        "override": {"inhibition_mode": "none"},
        "remove": [],
    },
    {
        "name": "inhib_fixed_all",
        "factor": "inhibition",
        "description": "Force fixed inhibition in all encoder layers",
        "override": {"inhibition_mode": "fixed", "inhibition_fixed_delta": 1.0},
        "remove": [],
    },
    {
        "name": "inhib_radial_all",
        "factor": "inhibition",
        "description": "Force radial inhibition in all encoder layers",
        "override": {"inhibition_mode": "radial", "inhibition_radial_mu": 1.0, "inhibition_radial_sigma": 1.0},
        "remove": [],
    },
    {
        "name": "enc_channels_1",
        "factor": "encoder_channels",
        "description": "Set all encoder layers to 1 channel",
        "override": {"encoder_channels_override": "1"},
        "remove": [],
    },
    {
        "name": "enc_channels_2",
        "factor": "encoder_channels",
        "description": "Set all encoder layers to 2 channels",
        "override": {"encoder_channels_override": "2"},
        "remove": [],
    },
    {
        "name": "enc_channels_4",
        "factor": "encoder_channels",
        "description": "Set all encoder layers to 4 channels",
        "override": {"encoder_channels_override": "4"},
        "remove": [],
    },
    {
        "name": "enc_layers_1",
        "factor": "encoder_depth",
        "description": "Keep only the first encoder layer",
        "override": {"encoder_layers_keep": 1},
        "remove": [],
    },
    {
        "name": "enc_layers_2",
        "factor": "encoder_depth",
        "description": "Keep only the first two encoder layers",
        "override": {"encoder_layers_keep": 2},
        "remove": [],
    },
    {
        "name": "dec_channels_8_4",
        "factor": "decoder_channels",
        "description": "Use decoder channels [8,4]",
        "override": {"channels": "8,4"},
        "remove": [],
    },
    {
        "name": "dec_channels_4",
        "factor": "decoder_channels",
        "description": "Use decoder channels [4]",
        "override": {"channels": "4"},
        "remove": [],
    },
    {
        "name": "decoder_pyramid",
        "factor": "decoder_type",
        "description": "Switch to pyramid decoder",
        "override": {"decoder_type": "pyramid"},
        "remove": ["channels"],
    },
    {
        "name": "latent_transformer",
        "factor": "latent_transformer",
        "description": "Add one lightweight Transformer encoder layer in the spatial bottleneck",
        "override": {
            "latent_transformer": True,
            "transformer_d_model": 32,
            "transformer_heads": 4,
            "transformer_layers": 1,
            "transformer_ff_dim": 64,
        },
        "remove": [],
    },
]


def _parse_int_list(value: str, default: List[int]) -> List[int]:
    if not value.strip():
        return list(default)
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_str_list(value: str, default: List[str]) -> List[str]:
    if not value.strip():
        return list(default)
    return [x.strip() for x in value.split(",") if x.strip()]


def _merge_params(base: Dict, override: Dict, remove: List[str]) -> Dict:
    out = dict(base)
    for k in remove:
        out.pop(k, None)
    out.update(override or {})
    return out


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)


def _mean_csv(path: Path, col: str) -> float:
    if not path.exists():
        return float("nan")
    vals = []
    with path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            v = row.get(col, "")
            if not v:
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
    return statistics.mean(vals) if vals else float("nan")


def _first_csv_float(path: Path, col: str) -> float:
    if not path.exists():
        return float("nan")
    with path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            v = row.get(col, "")
            if not v:
                continue
            try:
                return float(v)
            except ValueError:
                return float("nan")
    return float("nan")


def _pairwise_stats(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {
            "pair_auc_combo_mean": float("nan"),
            "pair_auc_combo_worst": float("nan"),
            "pair_auc_combo_worst_pos_mean": float("nan"),
            "pair_below_05": float("nan"),
        }
    rows = []
    with path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                pos = int(row["pos_class"])
                neg = int(row["neg_class"])
                auc_combo = float(row["auc_combo"])
            except Exception:
                continue
            rows.append((pos, neg, auc_combo))

    if not rows:
        return {
            "pair_auc_combo_mean": float("nan"),
            "pair_auc_combo_worst": float("nan"),
            "pair_auc_combo_worst_pos_mean": float("nan"),
            "pair_below_05": float("nan"),
        }

    vals = [r[2] for r in rows if r[2] == r[2]]
    if not vals:
        return {
            "pair_auc_combo_mean": float("nan"),
            "pair_auc_combo_worst": float("nan"),
            "pair_auc_combo_worst_pos_mean": float("nan"),
            "pair_below_05": float("nan"),
        }

    by_pos = defaultdict(list)
    for pos, neg, auc in rows:
        if auc == auc:
            by_pos[pos].append((neg, auc))

    worst_per_pos = []
    for pos, pairs in by_pos.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[1])
        if pairs_sorted:
            worst_per_pos.append(pairs_sorted[0][1])

    return {
        "pair_auc_combo_mean": float(statistics.mean(vals)),
        "pair_auc_combo_worst": float(min(vals)),
        "pair_auc_combo_worst_pos_mean": float(statistics.mean(worst_per_pos)) if worst_per_pos else float("nan"),
        "pair_below_05": float(sum(1 for v in vals if v < 0.5)),
    }


def _extract_worst_pairs(path: Path, dataset: str, variant: str, seed: int) -> List[Dict]:
    if not path.exists():
        return []
    by_pos = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                pos = int(row["pos_class"])
                neg = int(row["neg_class"])
                auc_combo = float(row["auc_combo"])
            except Exception:
                continue
            if auc_combo == auc_combo:
                by_pos[pos].append((neg, auc_combo))

    out = []
    for pos, vals in by_pos.items():
        vals = sorted(vals, key=lambda x: x[1])
        if not vals:
            continue
        neg, auc = vals[0]
        out.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seed": int(seed),
                "pos_class": int(pos),
                "worst_neg_class": int(neg),
                "worst_auc_combo": float(auc),
            }
        )
    return out


def _build_cmd(python_bin: str, config_path: Path, run_id: str, params: Dict, per_csv: Path, pair_csv: Path, tag: str) -> List[str]:
    cmd = [
        python_bin,
        str(Path(__file__).resolve().with_name("oneclass_auc_per_class.py")),
        "--config",
        str(config_path),
        "--csv_out",
        str(per_csv),
        "--pairwise_csv_out",
        str(pair_csv),
        "--ablation_tag",
        str(tag),
    ]
    if str(run_id).strip():
        cmd.extend(["--run_id", str(run_id).strip()])
    for k, v in params.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        elif v is None:
            continue
        else:
            cmd.extend([flag, str(v)])
    return cmd


def _dataset_spec(dataset: str, args) -> Dict[str, str]:
    if dataset not in DEFAULT_DATASET_SPECS:
        raise ValueError(f"Unsupported dataset '{dataset}'")
    spec = dict(DEFAULT_DATASET_SPECS[dataset])
    if dataset == "fashion_mnist":
        if args.fashion_config:
            spec["config"] = args.fashion_config
        if args.fashion_run_id:
            spec["run_id"] = args.fashion_run_id
    elif dataset == "mnist":
        if args.mnist_config:
            spec["config"] = args.mnist_config
        if args.mnist_run_id:
            spec["run_id"] = args.mnist_run_id
    elif dataset == "cifar10":
        if args.cifar10_config:
            spec["config"] = args.cifar10_config
        if args.cifar10_run_id:
            spec["run_id"] = args.cifar10_run_id
    return spec


def _materialize_config(repo_root: Path, src_rel: str, dst: Path, seed: int, class_ids: List[int], device: str) -> Path:
    src = (repo_root / src_rel).resolve()
    cfg = json.loads(src.read_text(encoding="utf-8"))
    cfg["seed"] = int(seed)
    cfg["device"] = str(device)
    cfg["class_ids"] = [int(c) for c in class_ids]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return dst


def _aggregate_summary(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        if int(row.get("return_code", 1)) != 0:
            continue
        grouped[(row["dataset"], row["variant"], row["factor"], row["description"])].append(row)

    out = []
    for (dataset, variant, factor, description), vals in sorted(grouped.items()):
        def _agg_num(key: str) -> Tuple[float, float]:
            xs = [float(v[key]) for v in vals if float(v[key]) == float(v[key])]
            if not xs:
                return float("nan"), float("nan")
            return float(statistics.mean(xs)), float(statistics.stdev(xs) if len(xs) > 1 else 0.0)

        auc_combo_mean, auc_combo_std = _agg_num("mean_auc_test_combo")
        auc_recon_mean, auc_recon_std = _agg_num("mean_auc_test_recon")
        pair_worst_pos_mean, pair_worst_pos_std = _agg_num("pair_auc_combo_worst_pos_mean")
        wall_mean, wall_std = _agg_num("wall_seconds")
        params_mean, params_std = _agg_num("params")
        param_bytes_mean, param_bytes_std = _agg_num("param_bytes")
        infer_flops_mean, infer_flops_std = _agg_num("infer_flops")

        out.append(
            {
                "dataset": dataset,
                "variant": variant,
                "factor": factor,
                "description": description,
                "n_runs": int(len(vals)),
                "mean_auc_test_combo": auc_combo_mean,
                "std_auc_test_combo": auc_combo_std,
                "mean_auc_test_recon": auc_recon_mean,
                "std_auc_test_recon": auc_recon_std,
                "mean_pair_auc_combo_worst_pos": pair_worst_pos_mean,
                "std_pair_auc_combo_worst_pos": pair_worst_pos_std,
                "mean_wall_seconds": wall_mean,
                "std_wall_seconds": wall_std,
                "mean_params": params_mean,
                "std_params": params_std,
                "mean_param_bytes": param_bytes_mean,
                "std_param_bytes": param_bytes_std,
                "mean_infer_flops": infer_flops_mean,
                "std_infer_flops": infer_flops_std,
            }
        )
    return out


def _aggregate_worst_errors(worst_rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in worst_rows:
        grouped[(row["dataset"], row["variant"], row["pos_class"], row["worst_neg_class"])].append(float(row["worst_auc_combo"]))

    out = []
    for (dataset, variant, pos_class, neg_class), vals in sorted(grouped.items()):
        out.append(
            {
                "dataset": dataset,
                "variant": variant,
                "pos_class": int(pos_class),
                "worst_neg_class": int(neg_class),
                "freq": int(len(vals)),
                "mean_worst_auc_combo": float(statistics.mean(vals)),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Detailed architecture ablation study for S4 reconstruction-only (no prototypes).")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--out_dir", default="runs/benchmark/ablations/s4_noproto")
    parser.add_argument("--datasets", default="fashion_mnist,mnist,cifar10")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--class_ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--device", default="cuda_if_available")
    parser.add_argument("--epochs_mult", type=int, default=2)
    parser.add_argument("--only", default="", help="Comma-separated ablation names")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--fashion_config", default="")
    parser.add_argument("--fashion_run_id", default="")
    parser.add_argument("--mnist_config", default="")
    parser.add_argument("--mnist_run_id", default="")
    parser.add_argument("--cifar10_config", default="")
    parser.add_argument("--cifar10_run_id", default="")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    python_bin = args.python_bin
    if not Path(python_bin).is_absolute():
        python_bin = str((repo_root / python_bin).resolve())

    datasets = _parse_str_list(args.datasets, ["fashion_mnist", "mnist", "cifar10"])
    seeds = _parse_int_list(args.seeds, [42])
    class_ids = _parse_int_list(args.class_ids, list(range(10)))

    ablations = list(ABLATIONS)
    if args.only.strip():
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        ablations = [a for a in ablations if a["name"] in wanted]
    if not ablations:
        raise SystemExit("No ablations selected.")

    rows = []
    worst_rows = []

    cfg_dir = out_dir / "generated_configs"
    cfg_by_dataset_seed = {}
    for dataset in datasets:
        spec = _dataset_spec(dataset, args)
        for seed in seeds:
            cfg_path = cfg_dir / f"cfg_{dataset}_seed{int(seed)}.json"
            cfg_by_dataset_seed[(dataset, int(seed))] = (
                _materialize_config(
                    repo_root,
                    spec["config"],
                    cfg_path,
                    seed=int(seed),
                    class_ids=class_ids,
                    device=args.device,
                ),
                spec["run_id"],
            )

    for dataset in datasets:
        for seed in seeds:
            cfg_path, run_id = cfg_by_dataset_seed[(dataset, int(seed))]
            for abl in ablations:
                name = str(abl["name"])
                params = _merge_params(
                    dict(BASE_RECON_ONLY, epochs_mult=int(args.epochs_mult)),
                    abl.get("override", {}),
                    abl.get("remove", []),
                )

                per_csv = out_dir / "per_class" / dataset / f"{name}_seed{int(seed)}.csv"
                pair_csv = out_dir / "pairwise" / dataset / f"{name}_seed{int(seed)}_pairwise.csv"
                log_file = out_dir / "logs" / dataset / f"{name}_seed{int(seed)}.log"

                cmd = _build_cmd(python_bin, cfg_path, run_id, params, per_csv, pair_csv, name)

                print(f"\\n=== dataset={dataset} seed={int(seed)} variant={name} ===")
                print(" ".join(shlex.quote(x) for x in cmd))

                if args.resume and per_csv.exists() and pair_csv.exists() and (not args.dry_run):
                    rc = 0
                    wall = 0.0
                    print("[resume] using existing outputs")
                elif args.dry_run:
                    rc = 0
                    wall = 0.0
                else:
                    per_csv.parent.mkdir(parents=True, exist_ok=True)
                    pair_csv.parent.mkdir(parents=True, exist_ok=True)
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                    if per_csv.exists():
                        per_csv.unlink()
                    if pair_csv.exists():
                        pair_csv.unlink()
                    if log_file.exists():
                        log_file.unlink()

                    t0 = time.perf_counter()
                    with log_file.open("w", encoding="utf-8") as lf:
                        proc = subprocess.run(
                            cmd,
                            cwd=str(repo_root),
                            stdout=lf,
                            stderr=subprocess.STDOUT,
                            text=True,
                            check=False,
                        )
                    wall = float(time.perf_counter() - t0)
                    rc = int(proc.returncode)

                mean_auc_combo = _mean_csv(per_csv, "auc_test_combo") if rc == 0 else float("nan")
                mean_auc_recon = _mean_csv(per_csv, "auc_test_recon") if rc == 0 else float("nan")
                mean_auc_latent = _mean_csv(per_csv, "auc_test_latent") if rc == 0 else float("nan")
                params_mean = _mean_csv(per_csv, "params") if rc == 0 else float("nan")
                param_bytes_mean = _mean_csv(per_csv, "param_bytes") if rc == 0 else float("nan")
                forward_flops_mean = _mean_csv(per_csv, "forward_flops") if rc == 0 else float("nan")
                infer_flops_mean = _mean_csv(per_csv, "infer_flops") if rc == 0 else float("nan")
                pstats = _pairwise_stats(pair_csv) if rc == 0 else {
                    "pair_auc_combo_mean": float("nan"),
                    "pair_auc_combo_worst": float("nan"),
                    "pair_auc_combo_worst_pos_mean": float("nan"),
                    "pair_below_05": float("nan"),
                }

                rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "variant": name,
                        "factor": str(abl["factor"]),
                        "description": str(abl["description"]),
                        "return_code": int(rc),
                        "wall_seconds": float(wall),
                        "mean_auc_test_combo": float(mean_auc_combo),
                        "mean_auc_test_recon": float(mean_auc_recon),
                        "mean_auc_test_latent": float(mean_auc_latent),
                        "pair_auc_combo_mean": float(pstats["pair_auc_combo_mean"]),
                        "pair_auc_combo_worst": float(pstats["pair_auc_combo_worst"]),
                        "pair_auc_combo_worst_pos_mean": float(pstats["pair_auc_combo_worst_pos_mean"]),
                        "pair_below_05": float(pstats["pair_below_05"]),
                        "params": float(params_mean),
                        "param_bytes": float(param_bytes_mean),
                        "forward_flops": float(forward_flops_mean),
                        "infer_flops": float(infer_flops_mean),
                        "params_json": json.dumps(params, ensure_ascii=True, sort_keys=True),
                        "per_class_csv": str(per_csv),
                        "pairwise_csv": str(pair_csv),
                        "log_file": str(log_file),
                        "command": " ".join(shlex.quote(x) for x in cmd),
                    }
                )

                if rc == 0:
                    worst_rows.extend(_extract_worst_pairs(pair_csv, dataset, name, int(seed)))
                    print(
                        "ok: combo={:.4f} recon={:.4f} worst-pos-pair={:.4f} wall={:.1f}s".format(
                            mean_auc_combo,
                            mean_auc_recon,
                            pstats["pair_auc_combo_worst_pos_mean"],
                            wall,
                        )
                    )
                else:
                    print(f"failed: rc={rc} log={log_file}")

    rows_sorted = sorted(rows, key=lambda r: (r["dataset"], r["variant"], r["seed"]))
    long_csv = out_dir / "s4_noproto_ablation_long.csv"
    _write_csv(
        long_csv,
        rows_sorted,
        [
            "dataset",
            "seed",
            "variant",
            "factor",
            "description",
            "return_code",
            "wall_seconds",
            "mean_auc_test_combo",
            "mean_auc_test_recon",
            "mean_auc_test_latent",
            "pair_auc_combo_mean",
            "pair_auc_combo_worst",
            "pair_auc_combo_worst_pos_mean",
                        "pair_below_05",
                        "params",
                        "param_bytes",
                        "forward_flops",
                        "infer_flops",
                        "params_json",
            "per_class_csv",
            "pairwise_csv",
            "log_file",
            "command",
        ],
    )

    summary_rows = _aggregate_summary(rows_sorted)
    summary_csv = out_dir / "s4_noproto_ablation_summary.csv"
    _write_csv(
        summary_csv,
        summary_rows,
        [
            "dataset",
            "variant",
            "factor",
            "description",
            "n_runs",
            "mean_auc_test_combo",
            "std_auc_test_combo",
            "mean_auc_test_recon",
            "std_auc_test_recon",
            "mean_pair_auc_combo_worst_pos",
            "std_pair_auc_combo_worst_pos",
            "mean_wall_seconds",
            "std_wall_seconds",
            "mean_params",
            "std_params",
            "mean_param_bytes",
            "std_param_bytes",
            "mean_infer_flops",
            "std_infer_flops",
        ],
    )

    worst_rows_sorted = sorted(worst_rows, key=lambda r: (r["dataset"], r["variant"], r["seed"], r["pos_class"]))
    worst_long_csv = out_dir / "s4_noproto_worst_pairs_long.csv"
    _write_csv(
        worst_long_csv,
        worst_rows_sorted,
        ["dataset", "variant", "seed", "pos_class", "worst_neg_class", "worst_auc_combo"],
    )

    agg_worst = _aggregate_worst_errors(worst_rows_sorted)
    agg_worst_csv = out_dir / "s4_noproto_worst_pairs_summary.csv"
    _write_csv(
        agg_worst_csv,
        agg_worst,
        ["dataset", "variant", "pos_class", "worst_neg_class", "freq", "mean_worst_auc_combo"],
    )

    md_path = out_dir / "README_s4_noproto_ablation.md"
    lines = [
        "# S4 No-Prototype Ablation",
        "",
        f"- Datasets: {','.join(datasets)}",
        f"- Seeds: {','.join(str(int(s)) for s in seeds)}",
        f"- Variants: {len(ablations)}",
        "",
        "## Outputs",
        f"- Long runs: `{long_csv}`",
        f"- Summary: `{summary_csv}`",
        f"- Worst-pair long: `{worst_long_csv}`",
        f"- Worst-pair summary: `{agg_worst_csv}`",
    ]

    by_dataset = defaultdict(list)
    for row in summary_rows:
        by_dataset[row["dataset"]].append(row)
    for dataset, vals in sorted(by_dataset.items()):
        vals = sorted(vals, key=lambda r: r["mean_auc_test_combo"], reverse=True)
        best = vals[0] if vals else None
        if best:
            lines.extend(
                [
                    "",
                    f"## {dataset}",
                    "",
                    "Top variant by mean AUC combo:",
                    f"- `{best['variant']}` -> {best['mean_auc_test_combo']:.4f}",
                ]
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved: {long_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {worst_long_csv}")
    print(f"Saved: {agg_worst_csv}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
