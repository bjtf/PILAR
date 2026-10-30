"""Reference checks for the paper's binary and region-level evaluators."""
from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from lipnet.metrics import pr_auc, roc_auc
from lipnet.pro_metrics import (
    aupro_at_fpr,
    clip_curve_at_fpr,
    connected_regions,
    thresholds_from_maps,
)


class BinaryMetricTests(unittest.TestCase):
    def assert_reference(self, scores: np.ndarray, labels: np.ndarray) -> None:
        self.assertAlmostEqual(roc_auc(scores, labels), roc_auc_score(labels, scores), places=14)
        self.assertAlmostEqual(pr_auc(scores, labels), average_precision_score(labels, scores), places=14)
        ranks = rankdata(scores, method="average")
        n_pos = labels.sum()
        rank_auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(labels) - n_pos))
        self.assertAlmostEqual(roc_auc(scores, labels), rank_auc, places=14)

    def test_perfect_and_inverse(self) -> None:
        labels = np.array([0, 0, 1, 1])
        for scores, expected in [(np.arange(4.0), 1.0), (-np.arange(4.0), 0.0)]:
            self.assert_reference(scores, labels)
            self.assertEqual(roc_auc(scores, labels), expected)

    def test_constant_scores_have_chance_auc_and_prevalence_ap(self) -> None:
        labels = np.array([0, 0, 1, 0, 1, 0, 0, 0])
        scores = np.ones(len(labels))
        self.assert_reference(scores, labels)
        self.assertEqual(roc_auc(scores, labels), 0.5)
        self.assertEqual(pr_auc(scores, labels), labels.mean())

    def test_ties_and_permutations_match_references(self) -> None:
        rng = np.random.default_rng(274)
        for n in (9, 53, 200):
            labels = np.arange(n) % 2
            scores = rng.integers(-3, 4, size=n).astype(float)
            expected = roc_auc(scores, labels), pr_auc(scores, labels)
            for _ in range(10):
                order = rng.permutation(n)
                self.assert_reference(scores[order], labels[order])
                self.assertEqual((roc_auc(scores[order], labels[order]), pr_auc(scores[order], labels[order])), expected)

    def test_untied_values_preserve_original_rank_calculation(self) -> None:
        rng = np.random.default_rng(42)
        for n in (4, 41, 1001):
            labels = np.arange(n) % 2
            scores = rng.permutation(n).astype(float)
            self.assert_reference(scores, labels)
            order = np.argsort(scores)
            ranks = np.empty(n, dtype=float)
            ranks[order] = np.arange(1, n + 1)
            n_pos = int(labels.sum())
            original_auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (n - n_pos))
            ordered_labels = labels[np.argsort(-scores)]
            tp = np.cumsum(ordered_labels)
            original_ap = np.sum(np.diff(np.r_[0, tp / n_pos]) * tp / np.arange(1, n + 1))
            self.assertEqual(roc_auc(scores, labels), original_auc)
            self.assertAlmostEqual(pr_auc(scores, labels), original_ap, places=14)

    def test_undefined_single_class_is_explicit(self) -> None:
        self.assertTrue(np.isnan(roc_auc([1, 2], [0, 0])))
        self.assertTrue(np.isnan(roc_auc([1, 2], [1, 1])))
        self.assertTrue(np.isnan(pr_auc([1, 2], [0, 0])))
        self.assertEqual(pr_auc([1, 2], [1, 1]), 1.0)

    def test_invalid_inputs_raise_instead_of_changing_labels(self) -> None:
        cases = [([], []), ([1], [0, 1]), ([np.nan, 1], [0, 1]),
                 ([np.inf, 1], [0, 1]), ([-np.inf, 1], [0, 1]),
                 ([1, 2], [0.5, 1]), ([1, 2], [0, 2]),
                 ([1, 2], [np.nan, 1])]
        for scores, labels in cases:
            for metric in (roc_auc, pr_auc):
                with self.subTest(metric=metric.__name__, scores=scores, labels=labels):
                    with self.assertRaises(ValueError):
                        metric(scores, labels)


def scipy_aupro_reference(heats: list[np.ndarray], masks: list[np.ndarray], limit: float, n: int) -> float:
    """Independent region labeling and integration for the specified protocol."""
    heat_stack = np.stack(heats).astype(np.float32)
    mask_stack = np.stack(masks).astype(bool)
    labels = [ndimage.label(mask, structure=np.ones((3, 3), dtype=int)) for mask in mask_stack]
    region_sizes = [np.bincount(label.ravel())[1:] for label, _ in labels]
    n_regions = sum(count for _, count in labels)
    normal_pixels = (~mask_stack).sum()
    scores = heat_stack.ravel().astype(np.float64)
    lo, hi = scores.min(), scores.max()
    eps = max(1e-12, abs(hi - lo) * 1e-6, abs(hi) * 1e-9)
    thresholds = np.unique(np.r_[hi + eps, np.quantile(scores, 1 - np.linspace(0, 1, n)), lo - eps])[::-1]
    best_pro = {}
    for threshold in thresholds:
        predictions = heat_stack >= threshold
        fpr = np.count_nonzero(predictions & ~mask_stack) / normal_pixels
        overlap = 0.0
        for predicted, (label, _), sizes in zip(predictions, labels, region_sizes):
            counts = np.bincount(label.ravel(), weights=predicted.ravel(), minlength=len(sizes) + 1)[1:]
            overlap += np.sum(counts / sizes)
        best_pro[fpr] = max(best_pro.get(fpr, 0.0), overlap / n_regions)
    fpr = np.array(sorted(best_pro))
    pro = np.maximum.accumulate([best_pro[x] for x in fpr])
    # The fixtures include endpoints at FPR 0 and 1.
    within = fpr < limit
    x = np.r_[fpr[within], limit]
    y = np.r_[pro[within], np.interp(limit, fpr, pro)]
    return float(np.trapezoid(y, x) / limit)


class RegionMetricTests(unittest.TestCase):
    def test_components_match_scipy(self) -> None:
        mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
        for connectivity, expected in ((4, 3), (8, 1)):
            regions = connected_regions(mask, connectivity)
            structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
            labels, count = ndimage.label(mask, structure=structure)
            self.assertEqual(len(regions), expected)
            self.assertEqual(len(regions), count)
            actual = {frozenset(zip(ys.tolist(), xs.tolist())) for ys, xs in regions}
            reference = {frozenset(zip(*np.nonzero(labels == i))) for i in range(1, count + 1)}
            self.assertEqual(actual, reference)

    def test_aupro_against_scipy_region_reference(self) -> None:
        rng = np.random.default_rng(43)
        masks = [rng.random((11, 9)) > 0.8 for _ in range(3)]
        for tied in (True, False):
            heats = [rng.integers(0, 5, size=mask.shape).astype(np.float32) if tied
                     else rng.normal(size=mask.shape).astype(np.float32) for mask in masks]
            info, raw, clipped = aupro_at_fpr(heats, masks, 0.30, 300, 8)
            self.assertAlmostEqual(info["aupro"], scipy_aupro_reference(heats, masks, 0.30, 300), places=14)
            self.assertEqual(clipped[-1]["fpr"], 0.30)
            self.assertGreaterEqual(len(raw), 2)
            swapped, _, _ = aupro_at_fpr(heats[::-1], masks[::-1], 0.30, 300, 8)
            self.assertAlmostEqual(swapped["aupro"], info["aupro"], places=14)

    def test_perfect_constant_and_inverse_maps(self) -> None:
        mask = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], dtype=bool)
        inverse = np.array([[0.4, 0.5, 0.1], [0.6, 0.2, 0.3], [0.7, 0.8, 0.9]])
        for heat, expected in [(mask.astype(float), 1.0), (inverse, 0.0), (np.ones(mask.shape), 0.15)]:
            info, _, _ = aupro_at_fpr([heat], [mask], 0.3, 300, 8)
            self.assertAlmostEqual(info["aupro"], expected, places=14)

    def test_curve_duplicate_fpr_and_interpolated_limit(self) -> None:
        curve = [{"fpr": 0.0, "pro": 0.0}, {"fpr": 0.0, "pro": 0.4},
                 {"fpr": 0.2, "pro": 0.6}, {"fpr": 0.4, "pro": 0.8},
                 {"fpr": 1.0, "pro": 1.0}]
        clipped = clip_curve_at_fpr(curve, 0.3)
        np.testing.assert_allclose([row["fpr"] for row in clipped], [0.0, 0.2, 0.3])
        np.testing.assert_allclose([row["pro"] for row in clipped], [0.4, 0.6, 0.7])

    def test_undefined_region_cases(self) -> None:
        for masks in ([], [np.zeros((2, 2), dtype=bool)], [np.ones((2, 2), dtype=bool)]):
            info, raw, clipped = aupro_at_fpr([np.ones(mask.shape) for mask in masks], masks, 0.3, 300, 8)
            self.assertTrue(np.isnan(info["aupro"]))
            self.assertEqual((raw, clipped), ([], []))

    def test_invalid_region_inputs(self) -> None:
        heat = np.ones((2, 2))
        mask = np.array([[0, 1], [0, 0]])
        cases = [([heat], [], 0.3, 300, 8), ([heat], [mask[:1]], 0.3, 300, 8),
                 ([heat * np.nan], [mask], 0.3, 300, 8), ([heat], [mask * 255], 0.3, 300, 8),
                 ([heat], [mask], 0.0, 300, 8), ([heat], [mask], np.nan, 300, 8),
                 ([heat], [mask], 1.1, 300, 8), ([heat], [mask], 0.3, 1, 8),
                 ([heat], [mask], 0.3, 2.5, 8), ([heat], [mask], 0.3, 300, 6)]
        for args in cases:
            with self.subTest(arguments=args):
                with self.assertRaises(ValueError):
                    aupro_at_fpr(*args)
        with self.assertRaises(ValueError):
            thresholds_from_maps([np.array([0.0, np.nan])], 300)
        with self.assertRaises(ValueError):
            connected_regions(np.zeros((2, 2)), 6)


if __name__ == "__main__":
    unittest.main()
