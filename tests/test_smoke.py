from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from analysis.verify_reference_results import verify
from lipnet.inhibition import LateralInhibition
from lipnet.metrics import pr_auc, roc_auc
from lipnet.one_class import LIPNetAutoencoder


ROOT = Path(__file__).resolve().parents[1]


class MetricsTests(unittest.TestCase):
    def test_binary_metrics_rank_perfectly(self) -> None:
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(roc_auc(scores, labels), 1.0)
        self.assertAlmostEqual(pr_auc(scores, labels), 1.0)


class ModelTests(unittest.TestCase):
    def test_fixed_inhibition_preserves_shape(self) -> None:
        block = LateralInhibition(h=1, mode="fixed", delta=1.0)
        output = block(torch.ones(2, 5, 5))
        self.assertEqual(tuple(output.shape), (2, 5, 5))
        self.assertTrue(torch.isfinite(output).all())

    def test_autoencoder_preserves_image_shape(self) -> None:
        model = LIPNetAutoencoder(
            input_size=8,
            input_channels=1,
            layers_2d=[
                {
                    "rf": 3,
                    "overlap": 1,
                    "channels": 1,
                    "inhibition": {"enabled": True, "mode": "fixed", "h": 1},
                }
            ],
            decoder_hidden=[],
            decoder_cfg={"type": "pure_pyramid", "channels": [2], "use_upsample": True},
        )
        image = torch.rand(2, 1, 8, 8)
        self.assertEqual(tuple(model(image).shape), tuple(image.shape))

    def test_selected_trials_are_embedded(self) -> None:
        for filename in (
            "benchmark_pyramid.json",
            "benchmark_pyramid_fashion.json",
            "benchmark_pyramid_cifar10.json",
        ):
            config = json.loads((ROOT / "configs" / filename).read_text(encoding="utf-8"))
            self.assertIn("selected_trial", config)
            self.assertEqual(len(config["selected_trial"]["layers_2d"]), 3)


class ArtifactTests(unittest.TestCase):
    def test_reference_inventory(self) -> None:
        checked = verify()
        self.assertEqual(len(checked), 6)
        self.assertIn("19,435 image scores", checked[1])
        self.assertIn("365 subtype results", checked[-1])


if __name__ == "__main__":
    unittest.main()
