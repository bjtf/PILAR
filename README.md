# PILAR

Code and reference results for **Pyramidal Inhibitory Lightweight
Autoassociative Reconstruction for Cost-Constrained Visual Anomaly Detection**.

Published in [Discover Artificial Intelligence (2026)](https://doi.org/10.1007/s44163-026-02671-5).

PILAR learns to reconstruct normal images or frozen pretrained features using a
compact pyramidal encoder, lateral inhibition, a spatial bottleneck, and a
reconstruction decoder. Anomaly scores are reconstruction residuals.
Feature-PILAR-FCN uses ResNet18 in both training and inference; only the PILAR
heads are trained.

This repository reproduces the paper's MVTec AD and VisA experiments, full-category
MVTec ablations, and complementary image and RGB reconstruction experiments.
The Python package is named `lipnet`; its models implement PILAR.

## Installation

Use Python 3.12 for the paper environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-paper.txt -e '.[test]'
python -m unittest discover -s tests -v
python analysis/verify_reference_results.py
```

`requirements-paper.txt` records the versions used by the main industrial
experiments. `requirements.txt` provides the separately tested compatibility
environment. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for environment,
checkpoint, and evaluation details. The first feature experiment downloads
torchvision's ImageNet-1K V1 ResNet18 weights if they are not already cached.

## Industrial experiments

Obtain [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
and [VisA](https://github.com/amazon-science/spot-diff) under their original
terms and keep their official layouts:

```text
data/mvtec_ad/
  bottle/train/good/...
  bottle/test/good/...
  bottle/test/<defect>/...
  bottle/ground_truth/<defect>/...
  ...
data/visa/
  split_csv/1cls.csv
  candle/Data/Images/Normal/...
  candle/Data/Images/Anomaly/...
  candle/Data/Masks/Anomaly/...
  ...
```

The main commands train ten epochs per category, with input size 512×512,
batch size 8, and five seeds. Training uses only normal images; evaluation uses
the final checkpoint and the official test split.

```bash
python scripts/reproduce_paper.py \
  --experiment mvtec --data-root data/mvtec_ad \
  --device mps --output-dir runs/mvtec

python scripts/reproduce_paper.py \
  --experiment visa --data-root data/visa \
  --device mps --output-dir runs/visa
```

Use `--device cuda` for an NVIDIA GPU or `--device cpu` for CPU execution.
Device selection is explicit. Results may vary across hardware and library
versions; the reference experiments used Apple MPS.

The MVTec component ablations cover all 15 categories at seed 42:

```bash
python scripts/reproduce_paper.py \
  --experiment mvtec-ablation --data-root data/mvtec_ad \
  --device mps --output-dir runs/mvtec_ablation
```

This trains four alternatives: no inhibition, no hard residual term, layer 2
only, and layer 3 only. Compare them with the seed-42 result from the main
MVTec campaign; that reference need not be trained a second time. There is no test-based checkpoint selection or
hyperparameter search in these commands.

Each run writes resolved configuration and provenance, complete-epoch training
checkpoints, a final inference export, per-image scores, evaluation maps, and
category metrics. The campaign also writes `category_metrics.csv`,
`image_scores.csv`, and `summary.json`. See the reproduction guide for resuming
an interrupted run and evaluating an export without training.

## Reference results

All values below are percentages, computed as a category macro within each
seed and then mean ± sample standard deviation over five seeds.

| Dataset | Image-AUROC | Image-AP | Pixel-AUROC | AUPRO@30 |
|---|---:|---:|---:|---:|
| MVTec AD | 96.91 ± 0.10 | 98.86 ± 0.04 | 96.49 ± 0.04 | 92.63 ± 0.05 |
| VisA | 92.53 ± 0.07 | 92.41 ± 0.13 | 97.84 ± 0.04 | 90.30 ± 0.14 |

[RESULTS.md](RESULTS.md) maps the experiments to the reference data, including
all category/seed metrics and 19,435 industrial image scores. The verifier
checks completeness and recomputes image AUROC/AP from these scores.

To regenerate screening curves and the 73-subtype MVTec analysis:

```bash
python analysis/analyze_mvtec_feature_pilar_subtype_review_budget.py \
  --output-dir results/generated/review_budget
```

To analyze a reproduced campaign, add `--scores runs/mvtec/image_scores.csv`
or the corresponding VisA file. Ranking ties use expected random ordering
within the tied block, without label-based tie breaking.

## Inference and timing

Inference loads the trained heads, ResNet weights, and normal-training score
statistics from a final export:

```python
import torch
from lipnet.inference import PilarDetector

checkpoint = (
    "runs/mvtec/mvtec/main/seed_42/details/wide64/hard_0p1x1/topk_0p01/"
    "model_bottle_layer2+layer3_hard0p1x1_zscore.pt"
)
model = PilarDetector(checkpoint, device="mps")
rgb = torch.rand(1, 3, 512, 512)  # CPU float32, RGB in [0, 1]
scores, maps = model.predict(rgb)
```

Measure the paper's PILAR timing protocol on completed seed-42 exports:

```bash
python scripts/benchmark_pilar.py \
  --run runs/mvtec/mvtec/main/seed_42 --data-root data/mvtec_ad \
  --output runs/timing --device mps
```

Run this without concurrent training. The timing scope starts at a resized
CPU RGB tensor and ends at the score and 256×256 map on CPU. It includes
normalization, transfers, feature extraction, reconstruction, and scoring;
it excludes disk reads and metric calculation. Hardware-dependent measurements
should be compared only under a shared protocol on the same device.

To regenerate the paper's fixed qualitative examples from the completed
seed-42 MVTec run:

```bash
python scripts/generate_mvtec_qualitative.py \
  --run runs/mvtec/mvtec/main/seed_42 --data_root data/mvtec_ad \
  --output_dir runs/qualitative
```

This uses the final evaluation maps and writes PNG/PDF figures and an example
selection manifest. It does not retrain a detector or select a different epoch.

## Complementary experiments

Commands and the fixed configurations for MNIST, Fashion-MNIST, CIFAR-10,
whole-image RGB, and patchwise RGB appear in
[docs/SUPPLEMENTARY_EXPERIMENTS.md](docs/SUPPLEMENTARY_EXPERIMENTS.md).
[docs/ARTIFACT_SCOPE.md](docs/ARTIFACT_SCOPE.md) describes the files and external
data dependencies.

## Citation

If you use PILAR or this code in your research, please cite the published article:

> Fernandes, B. J. T. Pyramidal inhibitory lightweight autoassociative reconstruction for cost-constrained visual anomaly detection. *Discover Artificial Intelligence* **6**, 1829 (2026). https://doi.org/10.1007/s44163-026-02671-5

```bibtex
@article{Fernandes2026PILAR,
  author  = {Fernandes, Bruno J. T.},
  title   = {Pyramidal inhibitory lightweight autoassociative reconstruction for cost-constrained visual anomaly detection},
  journal = {Discover Artificial Intelligence},
  year    = {2026},
  volume  = {6},
  number  = {1},
  pages   = {1829},
  doi     = {10.1007/s44163-026-02671-5},
  url     = {https://doi.org/10.1007/s44163-026-02671-5}
}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).
PILAR is distributed under the [MIT License](LICENSE).
