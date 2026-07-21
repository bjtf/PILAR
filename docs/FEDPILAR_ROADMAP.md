# FedPILAR research roadmap

## Recommended contribution

The next repository should be named **FedPILAR** and study **SP-FedPILAR:
Spatially Personalized Federated PILAR**. Merely applying FedAvg to the current
autoencoder would be too weak: lightweight federated anomaly detectors and
federated industrial visual models already exist.

The central research question is:

> Can structurally personalized, normal-only federation of Feature-PILAR heads
> transfer knowledge across non-IID industrial sites while preserving local
> camera geometry, approaching centralized detection quality, and reducing
> communication relative to full-head federation?

Pre-register three primary hypotheses:

1. SP-FedPILAR is non-inferior to centralized training within 2 AUPRO@30
   points under the selected protocol.
2. It improves macro and worst-client performance over local-only and full
   FedAvg under strong acquisition shift.
3. It reduces recurrent payload by at least 75% while keeping calibration
   statistics local.

## Structural split

PILAR layers contain spatially indexed `weight_map` and bias tensors. Those
parameters depend on camera alignment and resolution and are natural local
state. The 1×1 projections, inhibition parameters, decoder, and refinement are
spatially reusable and can be aggregated.

For the final two-head MVTec configuration at 512×512:

| State | Scope | Parameters |
|---|---:|---:|
| spatial weight maps and biases | local per client | 1,215,488 |
| projections, inhibition, decoder, refinement | federated | 332,294 |
| residual normalization and threshold | local per client | a few scalars |
| frozen ResNet18 | distributed once, never aggregated | 2,782,784 |

The full head is 1,547,782 parameters (6,191,128 bytes in fp32) per direction.
Federating only 332,294 parameters uses 1,329,176 bytes per direction, a 78.5%
reduction in recurrent payload. With five clients and 50 full-participation
rounds, bidirectional traffic falls from roughly 3.10 GB to 0.665 GB before
protocol overhead or compression.

Formally, client `k` maintains `(theta_shared, phi_k)`: both blocks are trained
locally, only the delta of `theta_shared` is sent, and `phi_k` persists locally.
A new client receives the shared block and adapts its spatial block and
calibration using only normal images.

## Experimental protocol

### Datasets and clients

- Fashion-MNIST and CIFAR-10: correction tests and inexpensive pilots only.
- MVTec AD: five simulated sites per category with disjoint normal training
  subsets, quantity skew, and fixed acquisition profiles (exposure, color
  temperature, blur/noise, compression). Anomalies remain test-only.
- Real-IAD: use five camera views as natural client proxies, with splits grouped
  by physical object instance to avoid cross-view leakage.
- MVTec AD 2: external lighting-shift and cold-start validation; follow its
  official validation and hidden-test protocol.

Simulated MVTec clients are not evidence of real factory collaboration, and
Real-IAD views are correlated. A genuinely multisite industrial dataset would
materially strengthen the paper.

### Heterogeneity

- quantity: balanced, Dirichlet alpha 0.5, and alpha 0.1;
- acquisition: camera/view/lighting profiles;
- system: client inputs at 256, 384, and 512 pixels and 1–5 local epochs;
- participation: 100% as the cross-silo primary case and 60% as stress;
- cold start: hold out a client and adapt with 5, 10, 25, or 50 normal images.

Start with five clients, 50 rounds, one local epoch, five paired seeds, and
equal sample-exposure/forward-pass budgets across methods. Split local normal
images into training and calibration sets before federation. Fit z-score or
conformal calibration only on local normal calibration data. Do not select
hyperparameters or stop training using anomalies.

### Baselines

- local-only PILAR;
- centralized pooled PILAR as a non-private upper bound;
- full-head FedAvg;
- full-head FedProx and SCAFFOLD;
- a generic personalized baseline such as Ditto or FedPer at the same shared
  parameter budget;
- SP-FedPILAR;
- SP-FedPILAR with int8/top-k update compression;
- a similarly sized federated convolutional autoencoder.

FedAvg is the canonical aggregation baseline, while SCAFFOLD targets client
drift and Ditto provides a strong personalization/fairness control:
[FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html),
[SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html), and
[Ditto](https://proceedings.mlr.press/v139/li21h).

### Metrics

Detection and fairness:

- Image-AUROC, Image-AP, Pixel-AUROC, and AUPRO@30;
- macro, worst-client, 10th-percentile, and between-client standard deviation;
- recall at review budget and recall/FPR at normal-only calibrated thresholds;
- negative transfer relative to each client's local-only model;
- held-out-client adaptation curves.

Cost:

- uplink, downlink, total bytes, and bytes to 95% of the best AUPRO;
- trainable/transmitted parameters and total inference state;
- training/inference FLOPs, measured latency, throughput, peak RAM, storage,
  and joules on fixed hardware.

Do not translate FLOPs directly into energy. Measure energy on a target such
as Raspberry Pi 5 or Jetson, record idle power, temperature, and throttling,
and report the measurement method.

Privacy:

- define an honest-but-curious server threat model;
- compare inversion and membership attacks on full versus partial updates;
- treat secure aggregation as a system control, not a formal privacy proof;
- if differential privacy is added, report clipping, noise, and explicit
  `(epsilon, delta)` accounting.

Federated learning without differential privacy does not justify a claim of
formal privacy because model updates can leak information.

### Statistics

- pair splits and seeds across methods;
- pre-register AUPRO@30 per communicated byte as the primary endpoint;
- use hierarchical bootstrap intervals by dataset/category/client, not pixel;
- use a pre-registered non-inferiority margin;
- apply Holm correction to secondary comparisons.

## Repository design

```text
FedPILAR/
  src/fedpilar/
    models/{pilar,inhibition,feature_pilar,parameter_partition}.py
    data/{mvtec,realiad,mvtec_ad2,client_splits,site_shifts}.py
    federated/{client,server,fedavg,fedprox,scaffold,compression}.py
    calibration/
    evaluation/
    experiments/
  configs/{model,datasets,experiments}/
  scripts/
  tests/
  docs/{protocol,threat-model,datasets,reproduction}.md
  artifacts/reference/
```

Keep JSON configuration to avoid an extra parser dependency. The highest-value
tests are: `K=1` equivalence, exact weighted aggregation, shared shapes across
resolutions, exact byte accounting, deterministic partitions, no anomalies in
training, and image/mask transform alignment.

## Milestones

1. Extract PILAR and establish numerical parity with this artifact.
2. Implement a deterministic simulator and `K=1/K=2` tests.
3. Validate communication accounting on Fashion-MNIST/CIFAR-10.
4. Run MVTec local/central/FedAvg/FedProx/SCAFFOLD controls.
5. Add the structural parameter split and local calibration.
6. Add Real-IAD, cold start, and heterogeneous resolutions.
7. Add compression, energy measurement, and privacy attacks.
8. Freeze the protocol, run five seeds, perform hierarchical analysis, and
   publish a versioned artifact.

## Main validity risks

- The frozen ResNet18 dominates the current 16.09-GFLOP inference path; claims
  must say “light trainable head/update,” not imply the full system is tiny.
- Keeping 78.5% of the head local lowers communication but increases onboarding
  state and can limit transfer.
- Local calibration is unstable with few normal samples.
- Partial updates and quantization are not privacy guarantees.
- A single-machine simulator does not validate network behavior, stragglers,
  or real energy.
- Spatial personalization overlaps conceptually with generic personalized FL;
  equal-budget baselines and parameter-group ablations are essential.

The official [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)
protocol and [Real-IAD](https://realiad4ad.github.io/Real-IAD/) multi-view data
are the strongest public starting points for realistic acquisition shift.
