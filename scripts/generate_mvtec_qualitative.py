#!/usr/bin/env python3
"""Render the paper's fixed examples from final seed-42 artifacts; no training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from PIL import Image

from lipnet.checkpoint import sha256
from scripts.run_mvtec_patch_lipnet import _find_mask, _find_mvtec_root, _image_to_tensor, _load_mask, _load_rgb

# These image identities were fixed before reevaluating the revised seed 42.
# Their selection does not depend on the new run's rankings.
PAPER_EXAMPLES = [
    ("bottle", "test/contamination/007.png"),
    ("leather", "test/glue/000.png"),
    ("hazelnut", "test/hole/010.png"),
    ("toothbrush", "test/defective/017.png"),
    ("zipper", "test/fabric_border/005.png"),
    ("screw", "test/scratch_neck/003.png"),
    ("screw", "test/thread_top/021.png"),
    ("screw", "test/good/022.png"),
]


def _unique_artifact(run: Path, pattern: str) -> Path:
    found = sorted(run.rglob(pattern))
    if len(found) != 1:
        raise ValueError(f"Expected one seed-42 artifact matching {pattern} under {run}; found {len(found)}")
    return found[0]


def _read_map(path: Path, image: str):
    with np.load(path, allow_pickle=False) as data:
        indices = np.flatnonzero(data["images"] == image)
        if len(indices) != 1:
            raise ValueError(f"Fixed image is missing or duplicated in {path}: {image}")
        index = int(indices[0])
        heat = data["heat_maps"][index].copy()
        mask = data["masks"][index].copy()
        score = float(data["scores"][index])
    if heat.ndim != 2 or heat.shape != mask.shape or not np.isfinite(heat).all() or not np.isfinite(score):
        raise ValueError(f"Invalid map/mask/score in {path}")
    return heat, mask, score


def _validate_checkpoint(path: Path):
    import torch
    # This is the locally generated export, not an externally downloaded pickle.
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if (state.get("method") != "pilar" or int(cfg["seed"]) != 42
            or int(cfg["epochs"]) != 10 or int(cfg["image_resize"]) != 512
            or str(cfg["feature_encoder_channels"]) != "64,64"
            or set(state["models"]) != {"layer2", "layer3"}
            or any(value["mode"] != "zscore" for value in state["stats"].values())):
        raise ValueError("Paper figures require the main wide64/layer2+layer3/zscore, seed-42, 10-epoch, 512-pixel export")


def _records(run: Path, data_root: Path, source: str, device: str):
    records = []
    detectors = {}
    digests = {}
    checked = set()
    for category, image in PAPER_EXAMPLES:
        image_path = data_root / category / image
        checkpoint = _unique_artifact(run, f"model_{category}_*.pt")
        if checkpoint not in checked:
            _validate_checkpoint(checkpoint)
            checked.add(checkpoint)
        if source == "maps":
            artifact = _unique_artifact(run, f"maps_{category}_*.npz")
            heat, mask, score = _read_map(artifact, image)
        else:
            from lipnet.inference import PilarDetector
            if category not in detectors:
                artifact = checkpoint
                detector = PilarDetector(artifact, device=device)
                detectors[category] = (detector, artifact)
            detector, artifact = detectors[category]
            rgb = _image_to_tensor(_load_rgb(image_path, int(detector.args.image_resize))).unsqueeze(0)
            scores, maps = detector.predict(rgb)
            score, heat = float(scores[0]), maps[0]
            mask_path = _find_mask(data_root / category, Path(image).parent.name, image_path)
            mask = _load_mask(mask_path, (heat.shape[1], heat.shape[0]))
        rgb = np.asarray(_load_rgb(image_path, 256), dtype=np.float32) / 255.0
        if artifact not in digests:
            digests[artifact] = sha256(artifact)
        heat = np.asarray(heat, dtype=np.float32)
        normalized = (heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-12)
        records.append({"category": category, "image": image, "rgb": rgb,
                        "heat": normalized, "mask": mask, "score": score,
                        "artifact": str(artifact.relative_to(run)), "artifact_sha256": digests[artifact],
                        "image_sha256": sha256(image_path)})
    return records


def _render(records, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    for name, selected in (("good", records[:5]), ("screw", [records[7], records[6], records[5]])):
        fig, axes = plt.subplots(len(selected), 4, figsize=(5.15, 1.06 * len(selected) + 0.4), squeeze=False)
        fig.subplots_adjust(left=0.13, right=0.997, top=0.945, bottom=0.025, wspace=0.025, hspace=0.075)
        for i, row in enumerate(selected):
            heat = np.asarray(Image.fromarray(row["heat"]).resize((256, 256), Image.Resampling.BILINEAR))
            overlay = 0.55 * row["rgb"] + 0.45 * plt.colormaps["inferno"](heat)[..., :3]
            axes[i, 0].imshow(row["rgb"])
            axes[i, 1].imshow(heat, cmap="inferno", vmin=0, vmax=1)
            axes[i, 2].imshow(overlay)
            axes[i, 3].imshow(row["mask"], cmap="gray", vmin=0, vmax=1)
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[:].set_visible(False)
            subtype = Path(row["image"]).parent.name.replace("_", " ")
            axes[i, 0].text(-0.08, 0.5, row["category"] + "\n" + subtype, ha="right", va="center",
                            rotation=90, transform=axes[i, 0].transAxes, fontsize=9)
        for ax, title in zip(axes[0], ("Input", "Score map", "Overlay", "Mask")):
            ax.set_title(title, fontsize=9, pad=5)
        fig.savefig(output / f"qualitative_{name}.pdf", dpi=300)
        fig.savefig(output / f"qualitative_{name}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="The complete MVTec main/seed_42 output directory")
    parser.add_argument("--data_root", default="data/mvtec_ad", type=Path)
    parser.add_argument("--output_dir", default="runs/mvtec_qualitative", type=Path)
    parser.add_argument("--source", choices=("maps", "checkpoints"), default="maps")
    parser.add_argument("--device", default="cpu", help="Only used with --source checkpoints")
    args = parser.parse_args()
    if not args.run.is_dir():
        raise FileNotFoundError(args.run)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    root = _find_mvtec_root(args.data_root)
    records = _records(args.run, root, args.source, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _render(records, args.output_dir)
    evidence = [{key: value for key, value in row.items() if key not in {"rgb", "heat", "mask"}} for row in records]
    manifest = {"seed": 42, "source": args.source, "selection": "Eight fixed paper image identities; no reselection by scores",
                "map_display": "Within-image min-max for visualization only; scores remain unchanged", "images": evidence}
    (args.output_dir / "qualitative-selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rendered eight fixed examples from final artifacts in {args.run}; output: {args.output_dir}")


if __name__ == "__main__":
    main()
