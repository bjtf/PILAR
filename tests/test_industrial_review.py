"""Regression cases from the independent industrial-protocol review."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from lipnet.checkpoint import sha256
from scripts.reproduce_paper import aggregate, METRICS, verify_run_artifacts
from scripts.run_mvtec_feature_lipnet_fcn_arch_sweep import _rng_state, _restore_rng


class IndustrialAggregationReviewTests(unittest.TestCase):
    def write_run(self, directory: Path, metric: float = 0.7) -> Path:
        directory.mkdir(parents=True)
        with (directory / "summary_arch_sweep.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["category", *METRICS])
            writer.writeheader()
            writer.writerow({"category": "bottle", **{name: metric for name in METRICS}})
        return directory

    def test_missing_seed_cannot_be_silently_zipped_away(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runs = [self.write_run(root / str(seed)) for seed in (42, 43)]
            with self.assertRaises(ValueError):
                aggregate({"main": runs}, ["bottle"], [42, 43, 44])

    def test_extra_seed_directory_cannot_be_silently_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runs = [self.write_run(root / str(seed)) for seed in (42, 43, 44)]
            with self.assertRaises(ValueError):
                aggregate({"main": runs}, ["bottle"], [42, 43])

    def test_probability_metric_range_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for value in (-0.1, 1.1):
                with self.subTest(value=value):
                    path = self.write_run(root / str(value), value)
                    with self.assertRaises(ValueError):
                        aggregate({"main": [path]}, ["bottle"], [42])

    def test_modified_summary_is_rejected_despite_valid_detail_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            run = self.write_run(Path(folder) / "seed_42")
            artifact = run / "details.csv"
            artifact.write_text("preserved detail artifact\n")
            ledger = run / "resume_state" / "bottle.json"
            ledger.parent.mkdir()
            ledger.write_text(json.dumps({
                "rows": [{"category": "bottle", **{name: 0.7 for name in METRICS}}],
                "artifacts": {str(artifact): sha256(artifact)},
            }))
            runs = {"main": [run]}
            verify_run_artifacts(runs, ["bottle"])
            summary = run / "summary_arch_sweep.csv"
            with summary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["category", *METRICS])
                writer.writeheader()
                writer.writerow({"category": "bottle", **{name: 0.9 for name in METRICS}})
            with self.assertRaises(ValueError):
                verify_run_artifacts(runs, ["bottle"])


class CategoryResumeReviewTests(unittest.TestCase):
    def test_json_category_cache_preserves_each_random_stream(self) -> None:
        device = torch.device("cpu")
        before = _rng_state(device)
        try:
            random.seed(43)
            np.random.seed(43)
            torch.manual_seed(43)
            # Include the optional cached Python/NumPy Gaussian values.
            random.gauss(0, 1)
            np.random.normal()
            checkpoint = json.loads(json.dumps(_rng_state(device)))
            expected = (random.gauss(0, 1), np.random.normal(size=5), torch.rand(5))
            _restore_rng(checkpoint, device)
            actual = (random.gauss(0, 1), np.random.normal(size=5), torch.rand(5))
            self.assertEqual(expected[0], actual[0])
            np.testing.assert_array_equal(expected[1], actual[1])
            torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
        finally:
            _restore_rng(before, device)


if __name__ == "__main__":
    unittest.main()
