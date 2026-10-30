from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lipnet.one_class import LIPNetAutoencoder
from scripts.ablation_s4_noproto_study import ABLATIONS, _validate_outputs, _materialize_config
from scripts.oneclass_auc_per_class import _recon_errors, _train_reconstruction
from scripts.run_cifar10_feature_lipnet_fcn import _summarize, _build_split, _run_one, _batch_score
from scripts.run_mvtec_feature_lipnet import _make_feature_model
from scripts.run_mvtec_feature_lipnet_fcn_sweep import LayerStats
from scripts.run_mvtec_patch_lipnet import _category_names, _find_mask, _patch_coords
from scripts.generate_mvtec_qualitative import PAPER_EXAMPLES, _read_map, _unique_artifact


ROOT = Path(__file__).resolve().parents[1]


class SupplementaryAggregationTests(unittest.TestCase):
    def rows(self):
        return [dict(seed=seed, normal_class=cls, feature_layers="layer2,layer3",
                     hard_loss="0.1:1", score_norm="zscore", auc=value, ap=value,
                     oracle_auc=value, inverted=0)
                for seed, values in ((42, [0.5, 0.9]), (43, [0.6, 1.0]))
                for cls, value in enumerate(values)]

    def test_sd_is_between_seed_macros_not_classes(self):
        result = _summarize(self.rows(), [0, 1], [42, 43])[0]
        self.assertAlmostEqual(result["mean"], 0.75)
        self.assertAlmostEqual(result["std"], np.std([0.7, 0.8], ddof=1))
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["n_classes"], 2)

    def test_incomplete_duplicate_and_nonfinite_matrices_rejected(self):
        rows = self.rows()
        for invalid in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                _summarize(invalid, [0, 1], [42, 43])
        rows[0]["auc"] = float("nan")
        with self.assertRaises(ValueError):
            _summarize(rows, [0, 1], [42, 43])

    def test_normal_only_training_split_and_complete_test(self):
        train = SimpleNamespace(targets=[0, 1, 0, 2, 1])
        test = SimpleNamespace(targets=[2, 0, 1, 0])
        split = _build_split(train, test, 0, ["zero", "one", "two"], 0, 0, np.random.default_rng(42))
        np.testing.assert_array_equal(split.train_idx, [0, 2])
        np.testing.assert_array_equal(split.test_idx, [0, 1, 2, 3])
        np.testing.assert_array_equal(split.test_labels, [1, 0, 1, 0])

    def test_feature_training_export_and_score_reload(self):
        class ToyDataset(TensorDataset):
            @property
            def targets(self):
                return self.tensors[1]

        class ToyBackbone(torch.nn.Module):
            def __init__(self, *unused):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 1)
                self.requires_grad_(False)

            def forward(self, x):
                return {"layer2": torch.nn.functional.avg_pool2d(self.conv(x), 2)}

        torch.manual_seed(42)
        train = ToyDataset(torch.rand(4, 3, 16, 16), torch.tensor([0, 1, 0, 1]))
        test = ToyDataset(torch.rand(4, 3, 16, 16), torch.tensor([0, 0, 1, 1]))
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(dataset="cifar10", max_train_images=0, max_test_images=0,
                                   batch_size=2, eval_batch_size=2, num_workers=0,
                                   backbone="resnet18", backbone_weights="none", epochs=1,
                                   lr=1e-3, weight_decay=0.0, feature_normalize=True,
                                   l1_weight=0.0, image_resize=16, image_agg="topk_mean",
                                   image_topk_frac=0.01, feature_encoder_channels="2,2",
                                   feature_rf_schedule="3,3", feature_overlap_schedule="2,2",
                                   feature_decoder_channels="4,2", inhibition_h=1,
                                   refine_decoder=True, score_norms=["zscore"],
                                   quantile_low=0.9, quantile_high=0.995, output_dir=temp)
            with patch("scripts.run_cifar10_feature_lipnet_fcn.ResNetMultiFeatureExtractor", ToyBackbone):
                rows = _run_one(train, test, 0, 42, ["layer2"], (0.1, 1.0), args, torch.device("cpu"))
            self.assertTrue(np.isfinite(rows[0]["auc"]))
            state = torch.load(Path(temp) / "model_seed42_class0_zscore.pt", weights_only=True)
            backbone = ToyBackbone().eval()
            backbone.load_state_dict(state["backbone"])
            models = {name: _make_feature_model(args, shape).eval()
                      for name, shape in state["feature_shapes"].items()}
            for name, model in models.items():
                model.load_state_dict(state["models"][name])
            stats = {name: LayerStats(**value) for name, value in state["stats"].items()}
            scores = np.concatenate([_batch_score(models, backbone, x, stats, torch.device("cpu"), args)
                                     for x, _ in DataLoader(test, batch_size=2)])
            with (Path(temp) / "scores_seed42_class0_zscore.csv").open() as stream:
                recorded = [float(row["score"]) for row in csv.DictReader(stream)]
            np.testing.assert_array_equal(scores, recorded)


class StrictProtocolTests(unittest.TestCase):
    def test_frozen_architectures_reproduce_parameter_counts(self):
        for filename, expected in (("benchmark_pyramid.json", 3670),
                                   ("benchmark_pyramid_fashion.json", 3670),
                                   ("benchmark_pyramid_cifar10.json", 13034)):
            cfg = json.loads((ROOT / "configs" / filename).read_text())
            trial = cfg["selected_trial"]
            decoder = dict(trial["decoder"], type="pure_pyramid", channels=[16, 8], use_upsample=False)
            model = LIPNetAutoencoder(input_size=cfg["model"]["input_size"],
                                     input_channels=cfg["model"].get("input_channels", 1),
                                     layers_2d=trial["layers_2d"], decoder_hidden=trial["decoder_hidden"],
                                     decoder_cfg=decoder)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
            self.assertNotIn("search_space", cfg)
        self.assertEqual(len(ABLATIONS), 7)

    def test_training_and_inference_export_preserve_scores(self):
        torch.manual_seed(42)
        kwargs = dict(input_size=8, input_channels=1,
                      layers_2d=[dict(rf=3, overlap=1, channels=1,
                                      inhibition=dict(enabled=True, mode="fixed", h=1))],
                      decoder_hidden=[], decoder_cfg=dict(type="pure_pyramid", channels=[2], use_upsample=True))
        model = LIPNetAutoencoder(**kwargs)
        loader = DataLoader(TensorDataset(torch.rand(4, 1, 8, 8), torch.zeros(4)), batch_size=2)
        before = _recon_errors(model, loader, torch.device("cpu"), "mse_l1", 0.2)
        _train_reconstruction(model, loader, torch.device("cpu"), 1, 1e-3, "mse_l1", 0.2)
        after = _recon_errors(model, loader, torch.device("cpu"), "mse_l1", 0.2)
        self.assertFalse(torch.equal(before, after))
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model.pt"
            torch.save(model.state_dict(), checkpoint)
            loaded = LIPNetAutoencoder(**kwargs)
            loaded.load_state_dict(torch.load(checkpoint, weights_only=True))
            torch.testing.assert_close(after, _recon_errors(loaded, loader, torch.device("cpu"), "mse_l1", 0.2), rtol=0, atol=0)

    def test_existing_configuration_cannot_change_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            _materialize_config(ROOT, "configs/benchmark_pyramid.json", path, 42, [0], "cpu")
            with self.assertRaises(FileExistsError):
                _materialize_config(ROOT, "configs/benchmark_pyramid.json", path, 43, [0], "cpu")

    def test_resume_rejects_partial_class_file(self):
        with tempfile.TemporaryDirectory() as temp:
            per = Path(temp) / "per.csv"
            with per.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["class_id"])
                writer.writeheader()
                writer.writerow({"class_id": 0})
            with self.assertRaises(ValueError):
                _validate_outputs(per, Path(temp) / "pair.csv", [0, 1], "mnist", 42, "baseline_recon_only")


class MVTecIntegrityTests(unittest.TestCase):
    def test_all_cannot_silently_mean_available_categories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "bottle/train/good").mkdir(parents=True)
            self.assertEqual(_category_names(root, "bottle"), ["bottle"])
            with self.assertRaises(FileNotFoundError):
                _category_names(root, "all")
            with self.assertRaises(ValueError):
                _category_names(root, "bottle,bottle")

    def test_missing_anomaly_mask_is_not_a_normal_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(FileNotFoundError):
                _find_mask(root, "scratch", root / "test/scratch/000.png")
            self.assertIsNone(_find_mask(root, "good", root / "test/good/000.png"))

    def test_patch_coverage_includes_edges(self):
        count = np.zeros((31, 35))
        for x, y in _patch_coords(35, 31, 8, 6):
            count[y:y+8, x:x+8] += 1
        self.assertTrue((count > 0).all())
        with self.assertRaises(ValueError):
            _patch_coords(35, 31, 8, 9)

    def test_qualitative_examples_are_fixed_and_maps_are_not_rescored(self):
        self.assertEqual(len(PAPER_EXAMPLES), 8)
        self.assertIn(("screw", "test/good/022.png"), PAPER_EXAMPLES)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "maps_bottle_final.npz"
            heat = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
            np.savez(path, images=["test/contamination/007.png"], heat_maps=heat,
                     masks=np.zeros_like(heat), scores=[123.45])
            loaded, _, score = _read_map(path, "test/contamination/007.png")
            np.testing.assert_array_equal(loaded, heat[0])
            self.assertEqual(score, 123.45)
            self.assertEqual(_unique_artifact(Path(temp), "maps_bottle_*.npz"), path)
            with self.assertRaises(ValueError):
                _read_map(path, "test/contamination/000.png")


if __name__ == "__main__":
    unittest.main()
