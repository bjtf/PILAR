# Reference results

Run `python analysis/verify_reference_results.py` to verify this table against
the tracked CSV files.

| Regime | Dataset | Main reference metric | Value | Artifact |
|---|---|---:|---:|---|
| Feature-PILAR-FCN | MVTec AD | Image-AUROC | 0.969487 | `results/reference/mvtec/feature_wide64/summary_arch_sweep.csv` |
| Feature-PILAR-FCN | MVTec AD | Image-AP | 0.988747 | same file |
| Feature-PILAR-FCN | MVTec AD | Pixel-AUROC | 0.965398 | same file |
| Feature-PILAR-FCN | MVTec AD | AUPRO@30 | 0.926461 | same file |
| whole-image RGB | MVTec AD | Image-AUROC | 0.695889 | `results/reference/mvtec/whole_rgb/summary_aupro.csv` |
| patchwise RGB | MVTec AD | Image-AUROC | 0.696582 | `results/reference/mvtec/patch_rgb/summary_aupro.csv` |
| Feature-PILAR-FCN | CIFAR-10 | one-vs-rest AUC | 0.812406 ± 0.077711 | `results/reference/cifar10/feature/summary.csv` |
| Feature-PILAR-FCN | Fashion-MNIST | one-vs-rest AUC | 0.936112 ± 0.046452 | `results/reference/fashion_mnist/feature/summary.csv` |
| Strict RGB PILAR | CIFAR-10 | one-vs-rest AUC | 0.573526 ± 0.117971 | `results/reference/strict/cifar10/summary_method_metrics.csv` |
| Strict grayscale PILAR | Fashion-MNIST | one-vs-rest AUC | 0.875989 ± 0.074775 | `results/reference/strict/mnist_fashion_mnist/summary_method_metrics.csv` |
| Strict grayscale PILAR | MNIST | one-vs-rest AUC | 0.946460 ± 0.050337 | same file |

The final MVTec system contains 2,782,784 frozen backbone parameters and
1,547,782 trainable PILAR-head parameters (4,330,566 total). The analytical
inference estimate is 16.0859 GFLOPs per 512×512 image; most of that cost is in
the frozen ResNet18 rather than the PILAR heads.

The subtype review-budget artifact contains 73 defect subtypes. Thirty-seven
reach the finite-sample oracle review fraction at 90% recall. Per-image score
files used to derive this analysis are included under
`results/reference/mvtec/feature_wide64/details`.

The 36-row feature-domain diagnostic over `screw` and `carpet` is tracked in
`results/reference/mvtec/feature_ablation/summary.csv`. Despite the historical
source-directory name, that completed result contains no `grid` category.
