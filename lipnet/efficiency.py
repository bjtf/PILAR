from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from lipnet.model import LIPNet2DLayer


def model_size_metrics(model: nn.Module) -> Dict[str, float]:
    params = [p for p in model.parameters() if p.requires_grad]
    return {
        "param_count": int(sum(p.numel() for p in params)),
        "param_bytes": int(sum(p.numel() * p.element_size() for p in params)),
    }


def _module_stage(name: str) -> str:
    if name.startswith("layers2d"):
        return "encoder"
    if name.startswith("decoder") or name in {"dec_fc", "pure_conv_out", "pure_upsample"}:
        return "decoder"
    return "other"


def _lipnet_layer_flops(module: LIPNet2DLayer, x: torch.Tensor) -> float:
    b = int(x.shape[0])
    h = int(x.shape[2])
    w = int(x.shape[3])
    c_out = int(module.channels)
    base = int(module.base_out_size)
    flops = 0.0

    if module.channel_proj is not None:
        in_c = int(x.shape[1])
        flops += 2.0 * b * c_out * h * w * in_c

    h_pad = h + int(module.pad_bottom)
    w_pad = w + int(module.pad_right)
    flops += 1.0 * b * c_out * h_pad * w_pad

    rf2 = int(module.rf) * int(module.rf)
    flops += 1.0 * b * c_out * base * base * rf2

    if module.norm_pre is not None:
        flops += 4.0 * b * c_out * base * base

    if module.inhibition is None:
        flops += 1.0 * b * c_out * base * base
    else:
        k = 2 * int(module.inhibition.h) + 1
        if base >= k:
            flops += 2.0 * b * c_out * base * base * (k * k)
        flops += 2.0 * b * c_out * base * base

    if module.norm_post is not None:
        flops += 4.0 * b * c_out * base * base

    if module.pool_enabled:
        out_s = int(module.out_size)
        k = int(module.pool_kernel)
        flops += 1.0 * b * c_out * out_s * out_s * max(0, k * k - 1)

    return flops


def _conv2d_flops(module: nn.Conv2d, y: torch.Tensor) -> float:
    b, out_c, out_h, out_w = map(int, y.shape)
    k_h, k_w = module.kernel_size
    in_c = int(module.in_channels)
    groups = int(module.groups)
    return 2.0 * b * out_c * out_h * out_w * (in_c / groups) * k_h * k_w


def _conv_t_flops(module: nn.ConvTranspose2d, y: torch.Tensor) -> float:
    b, out_c, out_h, out_w = map(int, y.shape)
    k_h, k_w = module.kernel_size
    in_c = int(module.in_channels)
    groups = int(module.groups)
    return 2.0 * b * out_c * out_h * out_w * (in_c / groups) * k_h * k_w


def _linear_flops(module: nn.Linear, x: torch.Tensor) -> float:
    b = int(x.shape[0])
    return 2.0 * b * int(module.in_features) * int(module.out_features)


def _upsample_flops(module: nn.Upsample, y: torch.Tensor) -> float:
    mode = str(getattr(module, "mode", "nearest"))
    return float(y.numel()) * (7.0 if mode in {"bilinear", "bicubic", "trilinear"} else 1.0)


def estimate_forward_flops_per_sample(model: nn.Module, input_shape: Tuple[int, int, int]) -> Dict[str, float]:
    model.eval()
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    totals = {"encoder": 0.0, "decoder": 0.0, "other": 0.0}
    hooks = []

    def _make_hook(name: str, kind: str):
        stage = _module_stage(name)

        def hook(module, inp, out):
            x = inp[0] if isinstance(inp, (tuple, list)) else inp
            y = out[0] if isinstance(out, (tuple, list)) else out
            if kind == "lipnet_layer":
                flops = _lipnet_layer_flops(module, x)
            elif kind == "conv2d":
                flops = _conv2d_flops(module, y)
            elif kind == "convtranspose2d":
                flops = _conv_t_flops(module, y)
            elif kind == "linear":
                flops = _linear_flops(module, x)
            elif kind == "activation":
                flops = float(y.numel())
            elif kind == "upsample":
                flops = _upsample_flops(module, y)
            elif kind == "maxpool":
                k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
                flops = float(y.numel()) * max(0, int(k) * int(k) - 1)
            elif kind == "layernorm":
                flops = float(y.numel()) * 5.0
            else:
                flops = 0.0
            totals[stage] = totals.get(stage, 0.0) + float(flops)

        return hook

    for name, module in model.named_modules():
        if not name:
            continue
        if name.startswith("layers2d") and isinstance(module, LIPNet2DLayer):
            hooks.append(module.register_forward_hook(_make_hook(name, "lipnet_layer")))
        elif isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(_make_hook(name, "conv2d")))
        elif isinstance(module, nn.ConvTranspose2d):
            hooks.append(module.register_forward_hook(_make_hook(name, "convtranspose2d")))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(_make_hook(name, "linear")))
        elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid, nn.GELU)):
            hooks.append(module.register_forward_hook(_make_hook(name, "activation")))
        elif isinstance(module, nn.Upsample):
            hooks.append(module.register_forward_hook(_make_hook(name, "upsample")))
        elif isinstance(module, nn.MaxPool2d):
            hooks.append(module.register_forward_hook(_make_hook(name, "maxpool")))
        elif isinstance(module, nn.LayerNorm):
            hooks.append(module.register_forward_hook(_make_hook(name, "layernorm")))

    c, h, w = map(int, input_shape)
    x = torch.randn(1, c, h, w, device=model_device)
    with torch.no_grad():
        _ = model(x)

    for handle in hooks:
        handle.remove()

    enc = totals.get("encoder", 0.0)
    dec = totals.get("decoder", 0.0)
    oth = totals.get("other", 0.0)
    return {
        "flops_forward_encoder": float(enc),
        "flops_forward_decoder": float(dec),
        "flops_forward_other": float(oth),
        "flops_forward_total": float(enc + dec + oth),
    }
