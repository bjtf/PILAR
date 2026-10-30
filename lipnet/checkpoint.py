"""Atomic epoch checkpoints and final inference exports."""
from __future__ import annotations
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch

SCHEMA = "pilar-checkpoint-v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "device": str(device),
            "accelerator": torch.mps.get_rng_state() if device.type == "mps" else
            torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def restore_rng(state, device):
    if state["device"] != str(device):
        raise ValueError("Checkpoint device mismatch")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "mps":
        torch.mps.set_rng_state(state["accelerator"])
    elif device.type == "cuda":
        torch.cuda.set_rng_state_all(state["accelerator"])


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        torch.save(cpu_tree(payload), f)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def resume_epoch(models, optimizer, device, path, fingerprint, backbone=None):
    if not Path(path).exists():
        return 0
    # Only load artifacts written by this local workflow, never untrusted pickle files.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["schema"] != SCHEMA or state["fingerprint"] != fingerprint:
        raise ValueError("Epoch checkpoint provenance mismatch; use a new output directory")
    for layer, model in models.items():
        model.load_state_dict(state["models"][layer], strict=True)
    if backbone is not None:
        backbone.load_state_dict(state["backbone"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"], device)
    print(f" restored complete epoch {state['epoch']}", flush=True)
    return int(state["epoch"])


def save_epoch(models, optimizer, backbone, device, epoch, path, fingerprint):
    atomic_save({"schema": SCHEMA, "fingerprint": fingerprint, "epoch": epoch,
                 "models": {k: m.state_dict() for k, m in models.items()},
                 "backbone": backbone.state_dict(), "optimizer": optimizer.state_dict(),
                 "rng": rng_state(device)}, path)


def export_inference(models, backbone, stats, args, path):
    from dataclasses import asdict
    from lipnet.metrics import EVALUATOR_VERSION
    atomic_save({"schema": SCHEMA, "method": "pilar", "evaluator": EVALUATOR_VERSION,
                 "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 "backbone": backbone.state_dict(),
                 "models": {k: m.state_dict() for k, m in models.items()},
                 "stats": {k: asdict(v) for k, v in stats.items()},
                 "modules": {k: str(m) for k, m in models.items()}}, path)
