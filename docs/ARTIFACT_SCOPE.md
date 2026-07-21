# Artifact scope

## Inclusion rule

A file is included only when it is needed to train a reported PILAR variant,
recompute its metrics, audit an article-level aggregate, or understand the
planned federated extension.

## Included

- `lipnet/`: model, inhibition, one-class training support, metrics, AUPRO,
  utilities, and efficiency accounting;
- `configs/`: the three strict-image configurations and selected trials;
- `scripts/`: strict, feature-space, whole-image RGB, patchwise RGB, final
  MVTec, qualitative, and ablation entry points;
- `analysis/`: saved-score verification and subtype review-budget analysis;
- `results/reference/`: compact aggregate tables and the MVTec image scores
  needed for operational analyses;
- `tests/`: offline CPU checks.

## Excluded

- datasets and dataset archives;
- checkpoints and pretrained weights;
- the submitted PDF, LaTeX source, correspondence, and author metadata beyond
  the citation file;
- incomplete candidates, smoke runs, progress logs, and raw feature-score
  matrices that are not needed to verify a paper claim;
- unrelated classification, video, street-scene, DOTA, NSL-KDD, and historical
  federated experiments;
- local virtual environments, editor state, caches, and absolute paths.

## Before making the repository public

1. Confirm the venue's code and anonymity policy.
2. Confirm that the MIT license and copyright holder remain appropriate for
   the final release.
3. Create an immutable release tag and archive it with a DOI.
4. Add the accepted paper DOI and final bibliographic metadata to
   `CITATION.cff`.
5. Re-run the offline tests, reference verifier, and secret/path scan.
6. State which results are single-seed and which are five-seed.

Do not use a GitHub URL tied to an identifiable account as an anonymous
double-blind review artifact. Use the venue-approved anonymous archival route
if code must be shared before the review decision.
