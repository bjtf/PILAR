# PILAR

Minimal reproducibility artifact for **Pyramidal Inhibitory Lightweight
Autoassociative Reconstruction for Cost-Constrained Visual Anomaly
Detection**.

PILAR is a normal-only reconstruction detector built from a compact pyramidal
encoder, lateral inhibition, a spatial bottleneck, and a pure-pyramid decoder.
The artifact covers the four regimes evaluated in the paper:

- strict grayscale reconstruction on MNIST and Fashion-MNIST;
- strict RGB reconstruction on CIFAR-10 and MVTec AD;
- patchwise RGB reconstruction on MVTec AD;
- Feature-PILAR-FCN with a frozen ResNet18 and trainable PILAR heads.

The historical Python package and script names retain `lipnet` so that the
released code stays numerically aligned with the experiments. In the paper,
that implementation is named **PILAR**.

## What is included

- the complete model, inhibition, scoring, efficiency, and AUPRO code;
- the experiment entry points needed for every reported PILAR regime;
- selected architectures embedded in the configurations (no AutoML history
  is required);
- compact reference tables and image-level MVTec scores;
- a reference-result verifier and CPU smoke tests;
- a concrete research plan for the proposed federated extension, FedPILAR.

Datasets, pretrained ResNet weights, checkpoints, intermediate sweeps, the
submitted manuscript, and unrelated LIPNet experiments are intentionally not
included. See [docs/ARTIFACT_SCOPE.md](docs/ARTIFACT_SCOPE.md).

## Environment

The local packaging and verification environment used Python 3.9.6 with the
versions pinned in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python analysis/verify_reference_results.py
```

The commands above create an environment and install dependencies. They are
instructions for artifact users; this repository does not vendor packages.

The ImageNet ResNet18 weights are fetched by torchvision on the first feature
experiment unless they are already cached. Dataset downloads are controlled by
the individual scripts.

## Main MVTec AD reproduction

Place the dataset under `data/mvtec_ad` using the official layout:

```text
data/mvtec_ad/
  bottle/train/good/...
  bottle/test/good/...
  bottle/test/<defect>/...
  bottle/ground_truth/<defect>/...
  ...
```

Run the final Feature-PILAR-FCN configuration:

```bash
python scripts/run_mvtec_feature_lipnet_fcn_final_validation.py \
  --data_root data/mvtec_ad \
  --categories all \
  --device cuda_if_available \
  --output_dir runs/mvtec_feature_pilar_final \
  --candidates wide64 \
  --seed 42 \
  --epochs 10 \
  --batch_size 8 \
  --image_resize 512 \
  --pro_eval_size 256 \
  --fpr_limit 0.30 \
  --num_thresholds 300 \
  --no-resume
```

The expected macro values are Image-AUROC `0.969487`, Image-AP `0.988747`,
Pixel-AUROC `0.965398`, and AUPRO@30 `0.926461`. The complete reference row is
in `results/reference/mvtec/feature_wide64/summary_arch_sweep.csv`.

Run both RGB-only controls:

```bash
python scripts/run_mvtec_rgb_lipnet_aupro_models.py \
  --data_root data/mvtec_ad \
  --categories all \
  --device cuda_if_available \
  --seed 42 \
  --epochs 20 \
  --candidates whole_rgb,patch_rgb
```

## CIFAR-10 and Fashion-MNIST feature experiments

The full five-seed CIFAR-10 command is:

```bash
python scripts/run_cifar10_feature_lipnet_fcn.py \
  --dataset cifar10 \
  --data_root data \
  --output_dir runs/cifar10_feature_pilar \
  --classes 0,1,2,3,4,5,6,7,8,9 \
  --seeds 42,43,44,45,46 \
  --device cuda_if_available \
  --download \
  --image_resize 128 \
  --epochs 8 \
  --feature_layers layer2,layer3 \
  --hard_loss 0.10:1.0 \
  --score_norms zscore \
  --image_topk_frac 0.01
```

Replace `--dataset cifar10` with `--dataset fashion_mnist` for the Fashion-MNIST
feature experiment. Omit `--download` when datasets are already present or
when running in an offline environment.

## Strict image experiments and ablation

The selected AutoML trials are frozen in `configs/*.json`; the original search
history is not needed. To reproduce the seven strict-image variants reported
in the paper over Fashion-MNIST and MNIST:

```bash
python scripts/ablation_s4_noproto_study.py \
  --out_dir runs/strict_pilar \
  --datasets fashion_mnist,mnist \
  --seeds 42,43,44,45,46 \
  --class_ids 0,1,2,3,4,5,6,7,8,9 \
  --device cuda_if_available \
  --epochs_mult 2 \
  --only baseline_recon_only,inhib_none,inhib_radial_all,enc_channels_1,enc_channels_2,enc_channels_4,dec_channels_4
```

For the CIFAR-10 strict RGB baseline, use the same command with
`--datasets cifar10 --only baseline_recon_only`. Removing `--only` executes the
current extended ablation superset, which also contains variants not tabulated
in the submitted paper. These strict scripts download torchvision datasets
when absent.

The saved MVTec feature-domain diagnostic can be reproduced with:

```bash
python scripts/run_mvtec_feature_lipnet_fcn_sweep.py \
  --data_root data/mvtec_ad \
  --categories screw,carpet \
  --device cuda_if_available \
  --output_dir runs/mvtec_feature_ablation \
  --seed 42 \
  --image_resize 512 \
  --batch_size 8 \
  --epochs 10 \
  --feature_layer_sets layer3 layer2 layer2,layer3 \
  --score_norms none zscore quantile \
  --hard_loss_settings 0:0 0.10:1.0 \
  --eval_pixel_auc \
  --pixel_auc_size 256
```

## Reproducibility notes

Exact commands, expected outputs, hardware caveats, and the claim-to-artifact
map are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and
[RESULTS.md](RESULTS.md).

No open-source license has been selected yet. Until a license is added, all
rights are reserved; collaborators may inspect the private artifact, but a
license must be chosen before public release.
