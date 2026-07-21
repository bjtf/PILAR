# Reproducibility guide

## Scope and provenance

This is a curated artifact, not a snapshot of the full research workspace.
Code was copied from the LIPNet research repository at commit
`87cad2b5343ca08f3e7c5a4b8dc68019d985fa5f` and then adjusted only to remove
machine-specific paths and make the selected strict architectures
self-contained.

The source code retains the historical `LIPNet` class names. The mapping used
throughout the paper is:

| Paper name | Released implementation |
|---|---|
| PILAR encoder/decoder | `lipnet.one_class.LIPNetAutoencoder` |
| lateral inhibition | `lipnet.inhibition.LateralInhibition` |
| Strict PILAR | `scripts/oneclass_auc_per_class.py` |
| Feature-PILAR-FCN | `scripts/run_*feature_lipnet_fcn*.py` |

## Audited local environment

The environment used to package and test this artifact contained:

| Component | Version |
|---|---:|
| Python | 3.9.6 |
| PyTorch | 2.8.0 |
| torchvision | 0.23.0 |
| NumPy | 2.0.2 |
| SciPy | 1.13.1 |
| Pillow | 11.3.0 |
| Matplotlib | 3.9.4 |
| pandas | 2.3.3 |

The first five packages are sufficient for training and core evaluation.
Matplotlib and pandas are needed by the saved-score analysis scripts.
Historical strict-run metadata recorded a Python 3.12 executable, so the
interpreter used for every original experiment was not captured uniformly.
That provenance gap is preserved here rather than presenting Python 3.9.6 as
the sole original training environment.

## Determinism and expected variation

The scripts seed Python, NumPy, and PyTorch. Nevertheless, exact equality is
not guaranteed across CUDA, MPS, and CPU backends or across library versions.
Compare reproduced aggregate metrics with the reference CSVs, report the
hardware/backend, and treat small floating-point differences as expected.

The MVTec final run uses a single seed (`42`). CIFAR-10, Fashion-MNIST, and the
strict image experiments use seeds `42` through `46`. This asymmetry is part
of the submitted protocol and should not be hidden when reusing the artifact.

## Dataset and pretrained-weight boundaries

This repository does not redistribute MNIST, Fashion-MNIST, CIFAR-10, MVTec
AD, or ImageNet weights. Obtain each dataset under its original terms.
Torchvision manages MNIST/Fashion-MNIST/CIFAR-10 downloads. MVTec AD must be
placed manually at `data/mvtec_ad`.

Feature-PILAR-FCN uses frozen torchvision ResNet18 features from `layer2` and
`layer3`. The backbone is frozen but is part of the inference state and FLOP
count. Only the PILAR heads are trained.

## Protocol controls

- Training data contain only the class/category designated as normal.
- Feature normalization and residual z-score statistics are fitted on normal
  training data only.
- Feature training adds the hardest 10% residuals to the mean reconstruction
  loss with unit weight.
- MVTec image scores average the top 1% of the anomaly map.
- Pixel-AUROC and AUPRO@30 are evaluated at 256×256; training/inference input is
  512×512 for the final MVTec experiment.
- AUPRO uses the implementation in `lipnet/pro_metrics.py` and 300 thresholds
  for the final feature model.

## Verification levels

1. `python -m unittest discover -s tests -v` checks metrics, inhibition, model
   shapes, and frozen configuration availability without downloading data.
2. `python analysis/verify_reference_results.py` validates the released CSV
   inventory and the aggregate values quoted in the paper.
3. Dataset-backed commands in `README.md` retrain models and regenerate scores.

The artifact contains no checkpoints. This is intentional: full reproduction
retrains from seeds and configurations, while the compact reference scores let
readers audit the reported aggregation without rerunning training.

## Known limitations

- The architecture sweep and final MVTec result were evaluated on the same
  benchmark categories. The artifact preserves that protocol; future work
  should separate model selection from the final hidden test.
- MVTec uses one final training seed, so the released number is not a variance
  estimate.
- FLOPs are analytical estimates from the local counter, not measured energy.
- Runtime columns in reference files are hardware-specific and should not be
  compared across machines.
- The current analytical counter gives 770,405 strict-image inference FLOPs,
  while historical reference CSVs report 770,411. The six-operation accounting
  drift does not affect predictions, AUCs, or the 3,670-parameter count.
- External baseline values in the paper were taken from their publications and
  were not rerun under this codebase.
