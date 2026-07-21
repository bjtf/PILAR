from typing import Dict, List, Optional

import math
import torch
import torch.nn.functional as F
from torch import nn

from .model import LIPNet2DLayer, make_activation


class LIPNetAutoencoder(nn.Module):
    """Autoencoder: LIPNet-style encoder + decoder to reconstruct input."""

    def __init__(
        self,
        input_size: int,
        layers_2d: List[Dict],
        decoder_hidden: List[int],
        decoder_cfg: Optional[Dict] = None,
        input_channels: int = 1,
        output_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.input_channels = int(input_channels)
        if self.input_channels <= 0:
            raise ValueError("input_channels must be > 0")
        self.output_channels = int(output_channels) if output_channels is not None else self.input_channels
        if self.output_channels <= 0:
            raise ValueError("output_channels must be > 0")
        self.decoder_cfg = decoder_cfg or {"type": "mlp"}
        self.decoder_type = self.decoder_cfg.get("type", "mlp")

        self.layers2d = nn.ModuleList()
        size = self.input_size
        curr_channels = self.input_channels
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

        self.encoder_out_size = size
        self.encoder_out_channels = curr_channels

        # Optional lightweight Transformer in bottleneck (after encoder, before decoder).
        # Config example:
        # decoder.transformer = {
        #   "enabled": true, "d_model": 32, "nhead": 4, "num_layers": 1,
        #   "dim_feedforward": 64, "dropout": 0.0, "activation": "gelu", "norm_first": true
        # }
        self.use_latent_transformer = False
        self.transformer_in_proj: nn.Module = nn.Identity()
        self.transformer_out_proj: nn.Module = nn.Identity()
        self.latent_transformer: Optional[nn.Module] = None
        self.transformer_pos: Optional[nn.Parameter] = None

        tr_cfg = self.decoder_cfg.get("transformer", {}) or {}
        if bool(tr_cfg.get("enabled", False)):
            d_model = int(tr_cfg.get("d_model", curr_channels))
            nhead = int(tr_cfg.get("nhead", 4))
            num_layers = int(tr_cfg.get("num_layers", 1))
            ff_dim = int(tr_cfg.get("dim_feedforward", max(32, d_model * 2)))
            dropout = float(tr_cfg.get("dropout", 0.0))
            activation = tr_cfg.get("activation", "gelu")
            norm_first = bool(tr_cfg.get("norm_first", True))

            if d_model <= 0:
                raise ValueError("transformer d_model must be > 0")
            if nhead <= 0:
                raise ValueError("transformer nhead must be > 0")
            if d_model % nhead != 0:
                raise ValueError("transformer d_model must be divisible by nhead")
            if num_layers <= 0:
                raise ValueError("transformer num_layers must be > 0")
            if ff_dim <= 0:
                raise ValueError("transformer dim_feedforward must be > 0")

            self.use_latent_transformer = True
            self.transformer_d_model = d_model
            self.transformer_in_proj = nn.Conv2d(curr_channels, d_model, kernel_size=1) if curr_channels != d_model else nn.Identity()
            self.transformer_pos = nn.Parameter(torch.zeros(1, self.encoder_out_size * self.encoder_out_size, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=norm_first,
            )
            self.latent_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.transformer_out_proj = nn.Conv2d(d_model, curr_channels, kernel_size=1) if d_model != curr_channels else nn.Identity()
            nn.init.normal_(self.transformer_pos, mean=0.0, std=0.02)

        self.flatten = nn.Flatten()
        enc_dim = size * size * curr_channels

        if self.decoder_type == "mlp":
            dec_layers: List[nn.Module] = []
            prev = enc_dim
            for h in decoder_hidden:
                dec_layers.append(nn.Linear(prev, h))
                dec_layers.append(nn.ReLU())
                prev = h
            dec_layers.append(nn.Linear(prev, self.output_channels * self.input_size * self.input_size))
            self.decoder = nn.Sequential(*dec_layers)
        elif self.decoder_type == "pyramid":
            start_size = int(self.decoder_cfg.get("start_size", 7))
            start_channels = int(self.decoder_cfg.get("start_channels", 32))
            act_name = self.decoder_cfg.get("activation", "relu")
            if self.input_size % start_size != 0:
                raise ValueError("input_size must be divisible by start_size for pyramid decoder")
            ratio = self.input_size // start_size
            steps = int(round(math.log(ratio, 2))) if ratio > 0 else 0
            if 2 ** steps != ratio:
                raise ValueError("input_size/start_size must be power of 2 for pyramid decoder")
            ch_list = self.decoder_cfg.get("channels", [32, 16])
            if len(ch_list) < max(1, steps):
                ch_list = list(ch_list) + [ch_list[-1]] * (max(1, steps) - len(ch_list))

            self.dec_fc = nn.Linear(enc_dim, start_channels * start_size * start_size)
            dec_layers = []
            in_ch = start_channels
            for i in range(steps):
                out_ch = int(ch_list[i])
                dec_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
                dec_layers.append(make_activation(act_name))
                in_ch = out_ch
            dec_layers.append(nn.Conv2d(in_ch, self.output_channels, kernel_size=3, padding=1))
            self.decoder = nn.Sequential(*dec_layers)
        elif self.decoder_type == "pure_pyramid":
            start_size = size
            start_channels = curr_channels
            act_name = self.decoder_cfg.get("activation", "relu")
            ratio = float(self.input_size) / float(start_size) if start_size > 0 else 1.0
            overshoot = bool(self.decoder_cfg.get("overshoot", False))
            if ratio > 1.0:
                steps = int(math.ceil(math.log(ratio, 2))) if overshoot else int(math.floor(math.log(ratio, 2)))
            else:
                steps = 0
            self.pure_target_size = self.input_size
            self.pure_current_size = int(start_size * (2 ** steps)) if steps > 0 else int(start_size)
            cfg_use_upsample = bool(self.decoder_cfg.get("use_upsample", True)) or overshoot
            self.pure_use_upsample = cfg_use_upsample and (self.pure_current_size != self.pure_target_size)
            ch_list = self.decoder_cfg.get("channels", [32, 16])
            if len(ch_list) < max(1, steps):
                ch_list = list(ch_list) + [ch_list[-1]] * (max(1, steps) - len(ch_list))
            dec_layers = []
            in_ch = start_channels
            for i in range(steps):
                out_ch = int(ch_list[i])
                dec_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
                dec_layers.append(make_activation(act_name))
                in_ch = out_ch
            self.decoder = nn.Sequential(*dec_layers)
            self.pure_conv_out = nn.Conv2d(in_ch, self.output_channels, kernel_size=3, padding=1)
            self.pure_upsample = nn.Upsample(
                size=(self.pure_target_size, self.pure_target_size),
                mode=self.decoder_cfg.get("upsample_mode", "bilinear"),
                align_corners=False,
            )
            self.pure_refine: Optional[nn.Module] = None
            self.pure_refine_residual = False
            refine_cfg = self.decoder_cfg.get("refine", {}) or {}
            if refine_cfg.get("type", "none") == "depthwise":
                refine_act = refine_cfg.get("activation", act_name)
                self.pure_refine_residual = bool(refine_cfg.get("residual", True))
                self.pure_refine = nn.Sequential(
                    nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3, padding=1, groups=self.output_channels),
                    make_activation(refine_act),
                    nn.Conv2d(self.output_channels, self.output_channels, kernel_size=1),
                )
        else:
            raise ValueError("Unknown decoder type: {}".format(self.decoder_type))

    def _apply_latent_transformer(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_latent_transformer:
            return x

        x = self.transformer_in_proj(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, HW, C]

        if self.transformer_pos is not None:
            if int(self.transformer_pos.shape[1]) == int(tokens.shape[1]):
                tokens = tokens + self.transformer_pos
            else:
                pos2d = self.transformer_pos.transpose(1, 2).reshape(
                    1, c, self.encoder_out_size, self.encoder_out_size
                )
                pos2d = F.interpolate(pos2d, size=(h, w), mode="bilinear", align_corners=False)
                pos = pos2d.flatten(2).transpose(1, 2)
                tokens = tokens + pos

        tokens = self.latent_transformer(tokens)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        x = self.transformer_out_proj(x)
        return x

    def encode(self, x: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        for layer in self.layers2d:
            x = layer(x)
        x = self._apply_latent_transformer(x)
        if flatten:
            return x.view(x.size(0), -1)
        return x

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        z = self.encode(x, flatten=False)
        if self.decoder_type == "pure_pyramid":
            out = self.decoder(z)
            if self.pure_use_upsample:
                out = self.pure_upsample(out)
            out = self.pure_conv_out(out)
            if out.shape[-1] != self.pure_target_size:
                out = F.interpolate(
                    out,
                    size=(self.pure_target_size, self.pure_target_size),
                    mode="bilinear",
                    align_corners=False,
                )
            if self.pure_refine is not None:
                refined = self.pure_refine(out)
                out = out + refined if self.pure_refine_residual else refined
        else:
            z_flat = z.view(z.size(0), -1)
            if self.decoder_type == "mlp":
                out = self.decoder(z_flat)
                out = out.view(out.size(0), self.output_channels, self.input_size, self.input_size)
            elif self.decoder_type == "pyramid":
                start_size = int(self.decoder_cfg.get("start_size", 7))
                start_channels = int(self.decoder_cfg.get("start_channels", 32))
                out = self.dec_fc(z_flat)
                out = out.view(out.size(0), start_channels, start_size, start_size)
                out = self.decoder(out)
            else:
                raise ValueError("Unknown decoder type: {}".format(self.decoder_type))

        if return_latent:
            z_flat = z.view(z.size(0), -1)
            return out, z_flat
        return out
