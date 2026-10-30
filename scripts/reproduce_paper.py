#!/usr/bin/env python3
"""Train the fixed industrial PILAR configurations and aggregate complete runs."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lipnet.checkpoint import sha256
from lipnet.runtime import device_lock
from lipnet.data import visa_split, VISA_CATEGORIES
from scripts.run_mvtec_patch_lipnet import MVTEC_CATEGORIES, _find_mvtec_root, _test_images, _list_images

SEEDS = [42, 43, 44, 45, 46]
METRICS = ('image_auc', 'image_ap', 'pixel_auc', 'aupro_30')
ABLATIONS = {
    'no_inhibition': ('layer2,layer3', '0.10:1.0', False),
    'no_hard_loss': ('layer2,layer3', '0:0', True),
    'layer2_only': ('layer2', '0.10:1.0', True),
    'layer3_only': ('layer3', '0.10:1.0', True),
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def command(args, variant, seed, destination):
    layers, hard, inhibition = ABLATIONS.get(variant, ('layer2,layer3', '0.10:1.0', True))
    return [sys.executable, str(ROOT / 'scripts/run_mvtec_feature_lipnet_fcn_arch_sweep.py'),
            '--dataset', 'visa' if args.experiment == 'visa' else 'mvtec',
            '--data_root', str(args.data_root), '--categories', ','.join(args.categories),
            '--device', args.device, '--output_dir', str(destination), '--seed', str(seed),
            '--architecture_preset', 'none', '--architectures', 'wide64:64,64:3,3:2,2:128,64:true',
            '--epochs', '10', '--batch_size', '8', '--num_workers', '0', '--lr', '0.001', '--weight_decay', '0',
            '--backbone', 'resnet18', '--backbone_weights', 'default', '--image_resize', '512',
            '--feature_layers', layers, '--feature_normalize', '--inhibition_h', '1', '--refine_decoder',
            '--hard_loss_settings', hard, '--l1_weight', '0', '--score_norms', 'zscore',
            '--image_agg', 'topk_mean', '--image_topk_fracs', '0.01', '--pro_eval_size', '256',
            '--fpr_limit', '0.30', '--num_thresholds', '300', '--connectivity', '8', '--save_artifacts',
            '--feature_inhibition' if inhibition else '--no-feature_inhibition',
            '--resume' if args.resume else '--no-resume']


def aggregate(runs, categories, seeds):
    """Reject missing/duplicate categories; each seed contributes one category macro."""
    records, summaries = [], {}
    for variant, paths in runs.items():
        if len(paths) != len(seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("Incomplete/duplicate seed matrix")
        seed_means = []
        for seed, path in zip(seeds, paths):
            with (path / 'summary_arch_sweep.csv').open(newline='') as f:
                rows = [r for r in csv.DictReader(f) if r['category'] in categories]
            if len(rows) != len(categories) or {r['category'] for r in rows} != set(categories):
                raise ValueError(f'Incomplete/duplicate category matrix: {path}')
            means = {k: statistics.mean(float(r[k]) for r in rows) for k in METRICS}
            import math
            if not all(math.isfinite(float(r[k])) and 0 <= float(r[k]) <= 1 for r in rows for k in METRICS):
                raise ValueError(f'Non-finite metric: {path}')
            seed_means.append({'seed': seed, **means})
            records.extend({'variant': variant, 'seed': seed, 'category': r['category'],
                            **{k: float(r[k]) for k in METRICS}} for r in rows)
        summaries[variant] = {'per_seed_macro': seed_means, 'metrics': {
            k: {'mean': statistics.mean(r[k] for r in seed_means),
                'sample_sd': statistics.stdev(r[k] for r in seed_means) if len(seeds) > 1 else None}
            for k in METRICS}}
    return records, summaries


def verify_run_artifacts(runs, categories):
    """Verify the category completion ledger before accepting aggregate results."""
    for paths in runs.values():
        for path in paths:
            ledgers = list((path / 'resume_state').glob('*.json'))
            if len(ledgers) != len(categories):
                raise ValueError(f'Missing category completion ledgers: {path}')
            found = []
            expected_rows = {}
            for ledger in ledgers:
                state = json.loads(ledger.read_text())
                if not state.get('rows') or not state.get('artifacts'):
                    raise ValueError(f'Empty completion ledger: {ledger}')
                found.extend(row['category'] for row in state['rows'])
                expected_rows.update({row['category']: row for row in state['rows']})
                for name, digest in state['artifacts'].items():
                    artifact = Path(name)
                    if not artifact.is_file() or sha256(artifact) != digest:
                        raise ValueError(f'Missing/changed completed artifact: {artifact}')
            if sorted(found) != sorted(categories):
                raise ValueError(f'Incomplete/duplicate completed categories: {path}')
            with (path/'summary_arch_sweep.csv').open(newline='') as stream:
                summary_rows = [row for row in csv.DictReader(stream) if row['category'] in categories]
            for row in summary_rows:
                if any(float(row[k]) != float(expected_rows[row['category']][k]) for k in METRICS):
                    raise ValueError(f'Summary differs from its completed category records: {path}')


def validate_dataset(root, dataset, categories):
    """Check the official split before GPU work; content hashes are saved per run."""
    import numpy as np
    from PIL import Image
    data = visa_split(root) if dataset == 'visa' else {
        c: {'train': _list_images(root / c / 'train/good'), 'test': _test_images(root / c)} for c in categories}
    totals = {'train': 0, 'test': 0}
    seen = set()
    for c in categories:
        rows = data[c]
        if not rows['train'] or {x.label for x in rows['test']} != {0, 1}:
            raise ValueError(f'Incomplete train/test class: {c}')
        for path in [*rows['train'], *[x.path for x in rows['test']]]:
            if path.resolve() in seen:
                raise ValueError(f'Duplicate or overlapping split image: {path}')
            seen.add(path.resolve())
        for item in rows['test']:
            if item.label and item.mask_path is None:
                raise ValueError(f'Missing anomaly mask: {item.path}')
            if item.mask_path is not None:
                with Image.open(item.mask_path) as im:
                    mask = np.asarray(im)
                with Image.open(item.path) as im:
                    size = im.size
                # VisA masks encode the binary foreground as either 1 or 255.
                if mask.ndim != 2 or not np.isin(mask, [0, 1, 255]).all() or not mask.any():
                    raise ValueError(f'Invalid binary defect mask: {item.mask_path}')
                if mask.shape != (size[1], size[0]):
                    raise ValueError(f'Image/mask spatial mismatch: {item.path}')
        totals['train'] += len(rows['train']); totals['test'] += len(rows['test'])
    full = list(VISA_CATEGORIES) if dataset == 'visa' else MVTEC_CATEGORIES
    if set(categories) == set(full):
        expected = {'train': 8659, 'test': 2162} if dataset == 'visa' else {'train': 3629, 'test': 1725}
        if totals != expected:
            raise ValueError(f'Official dataset totals differ: {totals} vs {expected}')
    return totals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', choices=['mvtec', 'visa', 'mvtec-ablation'], required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], required=True)
    p.add_argument('--seeds', help='Default:42–46; the paired ablations use42 only')
    p.add_argument('--categories', default='all', help='A subset is a diagnostic, not a paper macro')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--evaluate-only', action='store_true', help='Aggregate completed training outputs without training')
    a = p.parse_args()
    dataset = 'visa' if a.experiment == 'visa' else 'mvtec'
    full = list(VISA_CATEGORIES) if dataset == 'visa' else MVTEC_CATEGORIES
    a.categories = full if a.categories == 'all' else a.categories.split(',')
    if not a.categories or len(set(a.categories)) != len(a.categories) or not set(a.categories) <= set(full):
        p.error('Unknown or duplicate categories')
    a.categories = sorted(a.categories)  # One sequential RNG stream per seed, as in the paper.
    seeds = ([42] if a.experiment == 'mvtec-ablation' else SEEDS) if a.seeds is None else [int(s) for s in a.seeds.split(',')]
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(SEEDS):
        p.error('Choose distinct paper seeds42–46')
    if a.experiment == 'mvtec-ablation' and seeds != [42]:
        p.error('The paired paper ablation uses seed42')
    a.data_root = a.data_root.resolve()
    a.output_dir = a.output_dir.resolve()
    variants = list(ABLATIONS) if a.experiment == 'mvtec-ablation' else ['main']
    runs = {v: [a.output_dir / a.experiment / v / f'seed_{s}' for s in seeds] for v in variants}
    if a.dry_run:
        for v, paths in runs.items():
            for seed, path in zip(seeds, paths):
                print(shlex.join(command(a, v, seed, path)))
        return
    import torch
    import torchvision
    import numpy as np
    from PIL import __version__ as pillow_version
    if not a.evaluate_only:
        if a.device == 'mps' and (not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1'):
            raise RuntimeError('Real MPS is required; CPU fallback is not enabled by this workflow')
        if a.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        if dataset == 'mvtec':
            a.data_root = _find_mvtec_root(a.data_root)
        totals = validate_dataset(a.data_root, dataset, a.categories)
    else:
        totals = None
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with device_lock('cpu' if a.evaluate_only else a.device), (a.output_dir / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = a.output_dir / 'manifest.json'
        old = None
        identity = {'experiment': a.experiment, 'categories': a.categories, 'seeds': seeds, 'device': a.device}
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old['identity'] != identity or (not a.resume and not a.evaluate_only):
                raise ValueError('Existing output/configuration: use --resume or a new directory')
        elif a.evaluate_only:
            raise ValueError('Missing run manifest')
        manifest = {'identity': identity, 'command': sys.argv, 'status': 'running', 'split_totals': totals,
                    'python': platform.python_version(), 'host': platform.node(), 'platform': platform.platform(),
                    'versions': {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'numpy': np.__version__, 'pillow': pillow_version},
                    'sources': {str(f.relative_to(ROOT)): sha256(f) for folder in ['lipnet', 'scripts'] for f in sorted((ROOT / folder).glob('*.py'))},
                    'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
        if not a.evaluate_only:
            if old is not None:
                if any(old[k] != manifest[k] for k in ('sources', 'versions', 'python', 'platform', 'host')):
                    raise ValueError('Resume code/environment mismatch; preserve this run and use a new output directory')
                manifest['attempts'] = old.get('attempts', []) + [{'started_utc': manifest['started_utc'], 'command': sys.argv}]
                manifest['started_utc'] = old['started_utc']
            else:
                manifest['attempts'] = [{'started_utc': manifest['started_utc'], 'command': sys.argv}]
            atomic_json(manifest_path, manifest)
            try:
                for v, paths in runs.items():
                    for seed, path in zip(seeds, paths):
                        cmd = command(a, v, seed, path)
                        log = a.output_dir / 'logs' / f'{v}-{seed}-{time.time_ns()}.log'
                        log.parent.mkdir(exist_ok=True)
                        print(shlex.join(cmd), flush=True)
                        with log.open('w') as out:
                            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
                                for line in child.stdout:
                                    print(line, end='', flush=True); out.write(line); out.flush()
                                if child.wait():
                                    raise RuntimeError(f'Training failed; see {log}')
            except BaseException:
                manifest['status'] = 'interrupted_or_failed'
                atomic_json(manifest_path, manifest)
                raise
        verify_run_artifacts(runs, a.categories)
        records, summaries = aggregate(runs, a.categories, seeds)
        with (a.output_dir / 'category_metrics.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['variant', 'seed', 'category', *METRICS]); w.writeheader(); w.writerows(records)
        with (a.output_dir / 'image_scores.csv').open('w', newline='') as f:
            fields = ['dataset', 'seed', 'category', 'image', 'defect_type', 'label', 'score', 'variant']
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for v, paths in runs.items():
                for seed, path in zip(seeds, paths):
                    for category in a.categories:
                        files = list(path.glob(f'details/**/results_{category}_*.csv'))
                        if len(files) != 1:
                            raise ValueError(f'Missing/ambiguous scores for {category}')
                        with files[0].open() as score_file:
                            for row in csv.DictReader(score_file):
                                w.writerow({'dataset': dataset, 'seed': seed, 'variant': v, **{k: row[k] for k in fields if k in row}})
        full_run = a.categories == sorted(full) and seeds == ([42] if a.experiment == 'mvtec-ablation' else SEEDS)
        atomic_json(a.output_dir / 'summary.json', {'experiment': a.experiment, 'complete_paper_matrix': full_run,
                    'aggregation': 'Category macro within seed; arithmetic mean and sample SD across seeds. Single-seed SD is undefined.',
                    'categories': a.categories, 'seeds': seeds, 'results': summaries})
        if not a.evaluate_only:
            manifest['status'] = 'completed'; manifest['ended_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
            atomic_json(manifest_path, manifest)
        print(f'Complete paper matrix: {full_run}. Results: {a.output_dir / "summary.json"}')

if __name__ == '__main__':
    main()
