from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .inhibition import LateralInhibition


def make_activation(name: str) -> nn.Module:
    """Factory for activation modules."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    raise ValueError(f"Unknown activation: {name}")


def make_norm(norm_cfg: Optional[Dict], channels: int) -> Optional[nn.Module]:
    if not norm_cfg:
        return None

    norm_type = norm_cfg.get("type", "bn").lower()
    if norm_type == "bn":
        return nn.BatchNorm2d(channels)
    if norm_type == "gn":
        groups = int(norm_cfg.get("groups", 1))
        return nn.GroupNorm(groups, channels)
    if norm_type == "ln":
        return nn.LayerNorm([channels, norm_cfg.get("height", 1), norm_cfg.get("width", 1)])
    raise ValueError(f"Unknown norm type: {norm_type}")


class LIPNet2DLayer(nn.Module):
    """
    A single 2-D layer for LIPNet.

    Implements:
    - weight map tied to input positions
    - receptive fields with overlap
    - optional lateral inhibition
    - optional normalization before/after inhibition
    - optional pooling after activation
    - configurable activation (default ReLU)
    """

    def __init__(
        self,
        in_size: int,
        in_channels: int,
        channels: int,
        rf: int,
        overlap: int,
        activation: str = "relu",
        inhibition_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = None,
        pool_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.in_size = int(in_size)
        self.in_channels = int(in_channels)
        self.channels = int(channels)
        self.rf = int(rf)
        self.overlap = int(overlap)
        self.step = self.rf - self.overlap

        if self.step <= 0:
            raise ValueError("rf must be > overlap")

        numerator = (self.in_size - self.overlap)
        self.base_out_size = int((numerator + self.step - 1) // self.step)
        required_in = self.base_out_size * self.step + self.overlap
        self.pad_right = max(0, required_in - self.in_size)
        self.pad_bottom = max(0, required_in - self.in_size)

        self.channel_proj = None
        if self.in_channels != self.channels:
            self.channel_proj = nn.Conv2d(self.in_channels, self.channels, kernel_size=1, bias=False)

        self.weight_map = nn.Parameter(torch.randn(self.channels, 1, self.in_size, self.in_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(self.channels, self.base_out_size, self.base_out_size))

        self.activation = make_activation(activation)

        self.norm_cfg = norm_cfg or {}
        self.norm_position = self.norm_cfg.get("position", "none")
        if self.norm_cfg.get("type", "bn").lower() == "ln":
            self.norm_cfg["height"] = self.base_out_size
            self.norm_cfg["width"] = self.base_out_size

        self.norm_pre = make_norm(self.norm_cfg, self.channels) if self.norm_position in ["pre", "pre_post"] else None
        self.norm_post = make_norm(self.norm_cfg, self.channels) if self.norm_position in ["post", "pre_post"] else None

        self.inhibition_cfg = inhibition_cfg or {"enabled": False}
        self.inhibition: Optional[LateralInhibition] = None
        if self.inhibition_cfg.get("enabled", False):
            self.inhibition = LateralInhibition(
                h=int(self.inhibition_cfg.get("h", 1)),
                mode=self.inhibition_cfg.get("mode", "fixed"),
                delta=float(self.inhibition_cfg.get("delta", 1.0)),
                learnable_init=float(self.inhibition_cfg.get("learnable_init", 0.1)),
                radial_init_mu=float(self.inhibition_cfg.get("radial_init_mu", 1.5)),
                radial_init_sigma=float(self.inhibition_cfg.get("radial_init_sigma", 1.0)),
            )

        self.apply_to = self.inhibition_cfg.get("apply_to", "pre_activation")

        self.pool_cfg = pool_cfg or {"enabled": False}
        self.pool_enabled = bool(self.pool_cfg.get("enabled", False))
        self.pool_kernel = int(self.pool_cfg.get("kernel", 2))
        self.pool_stride = int(self.pool_cfg.get("stride", 2))

        if self.pool_enabled:
            pooled = (self.base_out_size - self.pool_kernel) // self.pool_stride + 1
            if pooled <= 0:
                raise ValueError("Pooling config invalid for current out_size")
            self.out_size = pooled
        else:
            self.out_size = self.base_out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_proj is not None:
            x = self.channel_proj(x)
        if x.shape[1] == 1 and self.channels > 1 and self.channel_proj is None:
            x = x.repeat(1, self.channels, 1, 1)
        elif x.shape[1] != self.channels:
            raise ValueError("Input channels do not match layer channels")

        if self.pad_right > 0 or self.pad_bottom > 0:
            x = F.pad(x, (0, self.pad_right, 0, self.pad_bottom))
            weight_map = F.pad(self.weight_map, (0, self.pad_right, 0, self.pad_bottom))
        else:
            weight_map = self.weight_map

        chis = []
        for c in range(self.channels):
            x_c = x[:, c : c + 1]
            w_c = weight_map[c : c + 1]
            x_weighted = x_c * w_c
            patches = F.unfold(x_weighted, kernel_size=self.rf, stride=self.step)
            chi = patches.sum(dim=1)
            b = x.shape[0]
            chi = chi.view(b, self.base_out_size, self.base_out_size) + self.bias[c]
            chis.append(chi)

        chi_all = torch.stack(chis, dim=1)

        if self.norm_pre is not None:
            chi_all = self.norm_pre(chi_all)

        if self.inhibition is None:
            out = self.activation(chi_all)
        else:
            outs = []
            for c in range(self.channels):
                chi_c = chi_all[:, c]
                if self.apply_to == "pre_activation":
                    inhib = self.inhibition(chi_c)
                    out_c = self.activation(chi_c - inhib)
                elif self.apply_to == "post_activation":
                    act = self.activation(chi_c)
                    inhib = self.inhibition(act)
                    out_c = act - inhib
                else:
                    raise ValueError("inhibition.apply_to must be pre_activation or post_activation")
                outs.append(out_c)
            out = torch.stack(outs, dim=1)

        if self.norm_post is not None:
            out = self.norm_post(out)

        if self.pool_enabled:
            out = F.max_pool2d(out, kernel_size=self.pool_kernel, stride=self.pool_stride)

        return out


class MLPClassifier(nn.Module):
    """Simple MLP head for 1-D layers."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_sizes: List[int],
        activation: str = "tanh",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        act = make_activation(activation)

        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LIPNet(nn.Module):
    """Full LIPNet model with multiple 2-D and 1-D layers."""

    def __init__(self, input_size: int, num_classes: int, layers_2d: List[Dict], layers_1d: Dict) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)
        self.concat_all_2d = bool(layers_1d.get("concat_all_2d", False))

        self.layers2d = nn.ModuleList()
        size = self.input_size
        sizes = []
        channels_list = []
        curr_channels = 1
        for layer_cfg in layers_2d:
            channels = int(layer_cfg.get("channels", curr_channels))
            layer = LIPNet2DLayer(
                in_size=size,
                in_channels=curr_channels,
                channels=channels,
                rf=int(layer_cfg["rf"]),
                overlap=int(layer_cfg["overlap"]),
                activation=layer_cfg.get("activation", "relu"),
                inhibition_cfg=layer_cfg.get("inhibition", {"enabled": False}),
                norm_cfg=layer_cfg.get("norm", None),
                pool_cfg=layer_cfg.get("pool", None),
            )
            self.layers2d.append(layer)
            size = layer.out_size
            curr_channels = channels
            sizes.append(size)
            channels_list.append(channels)

        self.flatten = nn.Flatten()
        if self.concat_all_2d:
            mlp_in = sum(sizes[i] * sizes[i] * channels_list[i] for i in range(len(sizes)))
        else:
            mlp_in = size * size * curr_channels
        self.mlp = MLPClassifier(
            input_dim=mlp_in,
            num_classes=self.num_classes,
            hidden_sizes=layers_1d.get("hidden_sizes", []),
            activation=layers_1d.get("activation", "tanh"),
            dropout=float(layers_1d.get("dropout", 0.0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.concat_all_2d:
            feats = []
            for layer in self.layers2d:
                x = layer(x)
                feats.append(self.flatten(x))
            x = torch.cat(feats, dim=1)
            return self.mlp(x)
        for layer in self.layers2d:
            x = layer(x)
        x = self.flatten(x)
        return self.mlp(x)
