# Complementary paper experiments

Run commands from the repository root after installation. These are the
image-domain controls and feature-space experiments reported alongside MVTec
and VisA. Each class/category is trained independently using only its normal
training images.

## Strict image reconstruction

MNIST and Fashion-MNIST use 28×28 grayscale inputs. The seven component
settings use the selected architecture embedded in `configs/`, a 40-epoch
budget (20 base epochs multiplied by `--epochs_mult 2`), and five seeds. The original architecture-search history is not
needed to train these fixed configurations.

```bash
python scripts/ablation_s4_noproto_study.py \
  --out_dir runs/strict_grayscale --data_root data \
  --datasets fashion_mnist,mnist --seeds 42,43,44,45,46 \
  --class_ids 0,1,2,3,4,5,6,7,8,9 --device mps \
  --epochs_mult 2 \
  --only baseline_recon_only,inhib_none,inhib_radial_all,enc_channels_1,enc_channels_2,enc_channels_4,dec_channels_4
```

The Strict RGB PILAR CIFAR-10 experiment uses 32×32 RGB inputs:

```bash
python scripts/ablation_s4_noproto_study.py \
  --out_dir runs/strict_cifar10 --data_root data \
  --datasets cifar10 --seeds 42,43,44,45,46 \
  --class_ids 0,1,2,3,4,5,6,7,8,9 --device mps \
  --epochs_mult 2 --only baseline_recon_only
```

Torchvision datasets are downloaded if absent; add `--no-download` for an
offline run. `--resume` reuses validated completed runs. It does not resume a
partially trained strict-image model from an intermediate epoch.

`per_class/` records the class metrics for each seed. The reported uncertainty
is the sample SD of the five ten-class macros, not the dispersion of all
class/seed observations. Strict-image reference data retain their original
class-level evaluation records; raw predictions for retrospective tie
correction were not retained.

## Feature reconstruction on CIFAR-10 and Fashion-MNIST

These experiments use frozen ResNet18 features, 128×128 RGB input, heads with
encoder channels `[32,32]`, decoder setting `[64,32]`, eight epochs, train batch
64 and evaluation batch 128. They are distinct from the wider 512×512
industrial setting.

```bash
python scripts/run_cifar10_feature_lipnet_fcn.py \
  --dataset cifar10 --data_root data \
  --output_dir runs/cifar10_feature_pilar \
  --classes 0,1,2,3,4,5,6,7,8,9 --seeds 42,43,44,45,46 \
  --device mps --download --image_resize 128 --epochs 8 \
  --batch_size 64 --eval_batch_size 128 \
  --feature_layers layer2,layer3 --hard_loss 0.10:1.0 \
  --score_norms zscore --image_topk_frac 0.01
```

For Fashion-MNIST, change `--dataset` to `fashion_mnist` and `--output_dir` to
`runs/fashion_mnist_feature_pilar`. Omit `--download` when the data are already
available. The grayscale images are expanded to three channels for ResNet18.
The script exports final models, image scores, and class metrics. Use a new
output directory for a fresh run; intermediate-epoch resumption is provided
by the industrial workflow, not by this script.

## MVTec RGB reconstruction-domain controls

```bash
python scripts/run_mvtec_rgb_lipnet_aupro_models.py \
  --data_root data/mvtec_ad --output_dir runs/mvtec_rgb_controls \
  --categories all --device mps --seed 42 --epochs 20 \
  --candidates whole_rgb,patch_rgb
```

Whole-image RGB reconstructs the 512×512 input with batch 8. Patchwise RGB
uses 128×128 patches with stride 64 from the 512×512 input, train batch 128,
and a maximum of 8,000 training patches. Both use residual maps, evaluation
maps of size 256×256, and 400 quantile thresholds for AUPRO@30. These RGB
controls therefore have a different threshold count from the 300-threshold
industrial feature protocol.

The wrapper writes separate `whole_rgb/` and `patch_rgb/` directories, including
per-image scores and final model exports. These are single-seed controls;
there is no between-seed SD to report. Use a fresh output directory for a new
training run. The published numerical reference preserves the original
control results; fresh results use the common tie-aware evaluator.
