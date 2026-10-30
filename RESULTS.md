# Reference results

`python analysis/verify_reference_results.py` verifies the tracked matrices and
recomputes the industrial image metrics from prediction scores. All metrics in
the data files use the range `[0,1]`; tables here show percentages.

## Main industrial results

Mean ± sample SD over five seed macros, each averaging all categories:

| Dataset | Categories | Image-AUROC | Image-AP | Pixel-AUROC | AUPRO@30 |
|---|---:|---:|---:|---:|---:|
| MVTec AD | 15 | 96.9078 ± 0.1050 | 98.8596 ± 0.0350 | 96.4908 ± 0.0369 | 92.6314 ± 0.0493 |
| VisA | 12 | 92.5316 ± 0.0745 | 92.4119 ± 0.1337 | 97.8359 ± 0.0427 | 90.2952 ± 0.1363 |

Both use input 512×512, ten training epochs and the final checkpoint.

## MVTec component ablations

All 15 categories, seed 42, ten epochs. Single-seed values have no between-seed
standard deviation. The main row reuses the seed-42 result from the main
experiment; it is not an additional independent repetition.

| Setting | Image-AUROC | Image-AP | Pixel-AUROC | AUPRO@30 |
|---|---:|---:|---:|---:|
| Main | 96.9524 | 98.8801 | 96.5388 | 92.6407 |
| No inhibition | 96.8295 | 98.7223 | 96.4845 | 92.7122 |
| No hard residual term | 94.0132 | 97.5447 | 95.7992 | 91.3764 |
| Layer 2 only | 92.7409 | 96.6304 | 91.4308 | 84.9494 |
| Layer 3 only | 96.5376 | 98.8142 | 96.4890 | 91.9226 |

## Complementary image experiments

Image-AUROC, category macro followed by mean ± sample SD over five seeds:

| Dataset | Strict image PILAR | Feature-PILAR-FCN |
|---|---:|---:|
| MNIST | 94.6460 ± 0.0568 | — |
| Fashion-MNIST | 87.5989 ± 0.2361 | 93.6112 ± 0.0661 |
| CIFAR-10 | 57.3526 ± 0.2455 | 81.2406 ± 0.0682 |

The strict grayscale component ablation contains seven settings, ten classes,
and five seeds for each of MNIST and Fashion-MNIST (700 class-level results).
MVTec whole-image and patchwise RGB controls have macro Image-AUROC 69.5889%
and 69.6582%, respectively, over 15 categories at seed 42.

Strict-image and RGB-control references preserve the original metric records
where predictions were not retained for retrospective correction. The feature
CIFAR-10/Fashion-MNIST values and industrial values use tie-aware AUROC/AP.
This distinction does not change the aggregation rule.

## Screening analysis

Review fraction needed to retrieve 90% of anomalies, mean ± sample SD:

| Analysis | Review fraction (%) |
|---|---:|
| MVTec, macro over 15 categories | 67.4615 ± 0.2451 |
| VisA, macro over 12 categories | 59.6871 ± 0.2636 |
| MVTec, macro over 73 defect subtypes | 38.6672 ± 0.1778 |

Subtype evaluation pairs each subtype with the normal test images from its own
category. Its prevalence differs from the full-category experiment; the two
review fractions are not the same operational task. Five seeds contribute to
each row. These test-set rankings characterize screening, not a calibrated
production rejection threshold.

## Files

Paths below are relative to `results/reference/`.

| File | Contents |
|---|---|
| `paper/industrial_category_metrics.csv` | 75 MVTec main + 60 VisA main + 60 MVTec ablation category results |
| `paper/industrial_summary.json` | Category identities, seed macros, mean and sample SD |
| `paper/image_scores.csv` | 19,435 scores: 1,725 MVTec and 2,162 VisA test images per seed |
| `paper/supplementary_class_metrics.csv` | 250 class results for the five complementary dataset/model combinations |
| `paper/supplementary_summary.json` | Correct seed-level summaries of those class results |
| `paper/strict_ablation_class_metrics.csv` | 700 strict grayscale component-ablation class results |
| `paper/strict_ablation_summary.json` | Strict ablation seed macros and uncertainty |
| `paper/mvtec/`, `paper/visa/` | Screening reference tables; MVTec includes 365 subtype/seed results |
| `paper/provenance.json` | Source identities, SHA-256 hashes, protocol and evaluation scope |
| `mvtec/whole_rgb/`, `mvtec/patch_rgb/` | RGB controls, per-category metrics and image scores |
| `strict/`, `cifar10/feature/`, `fashion_mnist/feature/` | Additional classwise and configuration details |

The full industrial detector retains 2,782,784 frozen backbone parameters and
1,547,782 trainable head parameters, totaling 4,330,566. FLOP and timing results
must state the operator scope, input/output boundary and device; repeat the
provided timing command on the device of interest. Literature baseline values
are outside this PILAR reproduction implementation.
