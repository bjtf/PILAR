# Reproducing the paper

## Experiments and fixed budgets

| Experiment | Categories | Seeds | Input | Epochs | Checkpoint |
|---|---:|---|---:|---:|---|
| Feature-PILAR-FCN, MVTec AD | 15 | 42–46 | 512×512 | 10 | final |
| Feature-PILAR-FCN, VisA | 12 | 42–46 | 512×512 | 10 | final |
| MVTec component ablations | 15 | 42 | 512×512 | 10 | final |
| Feature-PILAR-FCN, CIFAR-10/Fashion-MNIST | 10 per dataset | 42–46 | 128×128 | 8 | final |
| Strict image PILAR, MNIST/Fashion-MNIST/CIFAR-10 | 10 per dataset | 42–46 | 28×28 / 32×32 | 40 | final |
| Whole-image / patchwise RGB, MVTec AD | 15 | 42 | 512×512 / 128×128 patches from 512×512 | 20 | final |

The strict-image budget is 40 effective epochs: the stored base configuration
has 20 epochs, multiplied by `--epochs_mult 2`. The industrial and feature
budgets above are their effective epoch counts.

The industrial wrapper freezes these experimental choices. Use the complete
category lists for paper reproduction. A `--categories` subset is a diagnostic
run, not a substitute for the paper's macro result. Category ordering is
alphabetical and random state advances through a complete seed's sequence;
restarting a later category alone need not reproduce that category's original
initialization. Use complete-epoch checkpoint resumption when continuing a run.

The ablations alter only the selected component: inhibition, hard residual
loss, or feature-layer set. Their single-seed results describe paired
configuration differences, not uncertainty between seeds.

## Environment and provenance

Install `requirements-paper.txt` with Python 3.12 to match the industrial
software versions (PyTorch 2.10.0, torchvision 0.25.0, NumPy 2.4.2, SciPy 1.17.1,
and Pillow 12.1.0). The paper's PILAR industrial training used MPS on Apple
computers. `requirements.txt` pins a tested compatibility environment with
PyTorch 2.8.0 and torchvision 0.23.0; it must not be described as the original
training environment.

Python, NumPy, and PyTorch random generators are seeded. Backend kernels and
library versions can change floating-point results, so the code does not
promise identical training trajectories across CPU, MPS, and CUDA. Record the
backend and installed versions alongside reproduced results. A matching seed
alone does not make two differently configured runs identical.

Each industrial run records its resolved configuration, command, environment,
code/data identity, seeds, elapsed time, and checkpoint identity. Training
checkpoints preserve model, optimizer, and random state at completed epochs;
the final inference export also contains the frozen feature extractor and
normal-training normalization statistics. Checkpoints from an incompatible
configuration must not be reused.

## Industrial model and optimization

- Independent detectors are fitted for each category; all training images are
  normal. There is a separate PILAR head for each selected feature layer.
- The pretrained model is torchvision ResNet18 with `IMAGENET1K_V1` weights,
  frozen and in evaluation mode, including BatchNorm. Extraction ends at
  `layer3`. ImageNet channel normalization precedes extraction, and each
  spatial feature vector is L2-normalized across channels.
- Each head uses encoder channels `[64,64]`, receptive fields `[3,3]`, overlap
  `[2,2]` (stride 1), and ReLU. Its local weights and biases are spatially
  indexed; replacing these operators by a conventional convolution changes
  the model.
- The first stage has fixed inhibition with radius 1 and noncentral weights
  summing to 1. The second has radial inhibition with radius 1 and raw
  `(alpha, mu, sigma)` initialization `(0.1,1.0,1.0)`. The parameters undergo
  softplus; `mu` and `sigma` additionally receive `1e-6`. The center is masked.
  The external inhibition multiplier is implicitly 1; it is not a separate
  learned parameter.
- The industrial decoder bilinearly resizes the bottleneck, applies a padded
  3×3 output convolution, and adds a depthwise-3×3/ReLU/1×1 refinement residual.
  The constructor's `[128,64]` channel setting does not create transposed
  convolutions at these scale ratios.
- The loss sums the layer losses. For each layer, squared reconstruction
  residuals are averaged over channels; the spatial mean is augmented by the
  mean of the hardest 10% of locations in each image, with weight 1. L1 weight
  is zero.
- Optimization uses Adam, learning rate `1e-3`, weight decay 0, batch size 8,
  ten epochs, no scheduler, and no backbone updates.

## Data and scoring

MVTec AD uses the official normal training and test directories. VisA uses
`split_csv/1cls.csv`, exclusively normal training records, and the listed
image/mask pairs. Images are converted to RGB and directly bilinearly resized
to 512×512. There is no center crop or test-time augmentation. Normal test
images have all-zero masks; anomalous test images require their corresponding
binary masks. Train/test overlap and incomplete image/mask pairs are errors.

Residual mean and population standard deviation are fitted from normal
training data separately for each feature layer. Test residuals are z-scored,
bilinearly upsampled to input resolution (`align_corners=False`), and averaged
across layers. The image score is the mean of the highest 1% of values in this
512×512 map. Localization uses a bilinear resize to 256×256; binary masks use
nearest-neighbor resizing.

AUROC assigns half credit to tied positive/negative pairs. AP is
non-interpolated average precision at distinct score thresholds. AUPRO@30
uses 8-connected ground-truth regions, 300 quantile thresholds, interpolation
at false-positive rate 0.30, and division by 0.30. The quantile threshold
approximation is part of this protocol; alternative PRO implementations can
produce different values.

## Outputs, resumption, and inference

Under the requested output directory, results are separated by
`experiment/variant/seed_N/`. Category checkpoints, scores/maps, and resumption
state have separate locations. For example, a main MVTec run at
`--output-dir runs/mvtec` writes seed-42 inference exports under
`runs/mvtec/mvtec/main/seed_42/details/wide64/hard_0p1x1/topk_0p01/`, training
state under `checkpoints/`, and category-completion records under `resume_state/`
within that seed directory. Campaign `category_metrics.csv` and
`image_scores.csv` retain the category and seed identity. `summary.json`
aggregates complete category/seed matrices; incomplete diagnostics must not
be presented as the paper's final result.

Re-run the same industrial command and output directory with `--resume` to
continue a compatible interrupted execution. Complete-epoch state is used for resumption;
a partially executed epoch is repeated. Keep the same category list and
configuration. Copying only the model weights is insufficient to resume the
training trajectory.

Use `lipnet.inference.PilarDetector` to load a final inference export. Its
input is a CPU float32 RGB tensor in `[0,1]` at the configured resolution; its
outputs are image scores and evaluation maps. All backbone/head weights and
score-normalization statistics come from the export. A complete-epoch
training checkpoint is not interchangeable with this inference export.

To regenerate a category's scores, maps and all four metrics without training:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint runs/mvtec/mvtec/main/seed_42/details/wide64/hard_0p1x1/topk_0p01/model_bottle_layer2+layer3_hard0p1x1_zscore.pt \
  --data-root data/mvtec_ad --dataset mvtec --category bottle \
  --output-dir runs/bottle_evaluation --device mps
```

The evaluation directory must be new or empty. It receives `image_scores.csv`,
`maps.npz`, and `metrics.json`. This single-category evaluation does not replace
the complete campaign aggregation.

## Aggregation and screening

Compute all metrics per category first. Average categories within each seed;
report the mean and **sample standard deviation** (`ddof=1`) of those five
seed macros. Do not treat 50 class/seed pairs or 75 category/seed pairs as
independent repetitions. A category must not disappear from a final macro
because its run failed.

Review-budget curves follow the same category-then-seed aggregation. For
recall target `t`, `N` images and `m` anomalies, let `k=ceil(t*m)`. The oracle
review fraction is `k/N`, and expected random-order review fraction is
`k*(N+1)/(N*(m+1))`. Normalized efficiency is `(random-observed)/(random-oracle)`.
Within a tied block of `b` images containing `r` anomalies, the expected
position of the `j`th anomaly is `j*(b+1)/(r+1)`. No label-based ordering is
used to resolve ties. Each of the 73 MVTec subtypes is compared with the normal
images from its own category, independently for each seed.

## Timing

The timing entry point uses final seed-42 exports for all 15 categories in
FP32 evaluation mode. It starts with a resized CPU RGB tensor and includes
normalization, device transfer, backbone, head, scoring, and postprocessing
until score/map are back on CPU. Disk I/O and evaluation metrics are excluded.

- Latency: batch 1, 50 warmups, five blocks of 200 calls, with device
  synchronization before and after every timed call.
- Throughput: batch 16, 50 warmups, five blocks of 100 batches, synchronized at
  block boundaries; total images divided by elapsed time.
- Use identical image lists and no concurrent training. Keep temporal
  variability separate from variability between training seeds.

Parameter counts distinguish trainable heads from the complete retained
inference state. Report the arithmetic convention and executed operators with
FLOPs; they are not an estimate of energy or a hardware-independent runtime.

## Verification and reference boundaries

`python -m unittest discover -s tests -v` tests model and metric behavior without
requiring the benchmark datasets. `python analysis/verify_reference_results.py`
checks complete reference matrices, independently recalculates industrial
image AUROC/AP from scores, and verifies seed-level statistics. The screening
analysis recomputes the review-budget results from those scores.

Release validation loaded the original final seed-42 bottle checkpoint and
evaluated all 83 official bottle test images on MPS. A fresh ten-epoch run on
all 209 normal bottle training images was also completed. Both paths reproduced
the original 83 image scores, 83 maps at 256×256, and all four metrics exactly:
Image-AUROC `0.9976190476190476`, Image-AP `0.9992440151090305`, Pixel-AUROC
`0.9820492105915917`, and AUPRO@30 `0.9595557024492737`. This targeted check covers
training, export, reload and evaluation for one category. It does not claim
fresh retraining of all 195 industrial category/seed configurations or exact
equality across different backends.

The reference files distribute numerical results and image identities, not
dataset pixels, pretrained weights, or trained checkpoints. Training commands
produce checkpoints and pixel maps for fresh evaluation. Strict-image and RGB
control results retain their original metric records where raw predictions
were not retained; those records were not retrospectively corrected for ties.
CIFAR-10/Fashion-MNIST feature results and the industrial results use the
corrected evaluator. See [RESULTS.md](RESULTS.md) for the exact reference scope.
