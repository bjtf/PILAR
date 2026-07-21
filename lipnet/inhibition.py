from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class LateralInhibition(nn.Module):
    """
    Lateral inhibition block.

    Modes:
    - fixed:            psi is fixed (paper), based on delta and h
    - learnable_scalar: learn a single psi scalar
    - learnable_kernel: learn a full (2h+1)x(2h+1) kernel, center forced to 0
    - radial:           learn a ring-shaped Gaussian (center forced to 0)
    """

    def __init__(
        self,
        h: int,
        mode: str = "fixed",
        delta: float = 1.0,
        learnable_init: float = 0.1,
        radial_init_mu: float = 1.5,
        radial_init_sigma: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.h = int(h)
        self.mode = mode
        self.delta = float(delta)

        k = 2 * self.h + 1
        center = self.h

        # Mask used to zero the center (no self-inhibition).
        mask = torch.ones((k, k), dtype=torch.float32)
        mask[center, center] = 0.0
        self.register_buffer("mask", mask)

        if mode == "fixed":
            # psi from the paper: delta / ((2h+1)^2 - 1)
            denom = float((2 * self.h + 1) ** 2 - 1)
            self.register_buffer("psi", torch.tensor(self.delta / denom, dtype=torch.float32))
        elif mode == "learnable_scalar":
            # Softplus keeps psi >= 0.
            self.psi_raw = nn.Parameter(torch.tensor(learnable_init, dtype=torch.float32))
        elif mode == "learnable_kernel":
            # Raw kernel parameters (non-negative via softplus). Center is masked to 0.
            self.kernel_raw = nn.Parameter(torch.full((k, k), learnable_init, dtype=torch.float32))
        elif mode == "radial":
            # Learn a ring-shaped Gaussian: alpha * exp(-0.5 * ((d - mu)/sigma)^2), center forced to 0.
            # This yields stronger inhibition around distance ~mu and then decays.
            yy, xx = torch.meshgrid(
                torch.arange(k, dtype=torch.float32),
                torch.arange(k, dtype=torch.float32),
                indexing="ij",
            )
            dist = torch.sqrt((yy - center) ** 2 + (xx - center) ** 2)
            self.register_buffer("dist", dist)
            self.alpha_raw = nn.Parameter(torch.tensor(learnable_init, dtype=torch.float32))
            self.mu_raw = nn.Parameter(torch.tensor(radial_init_mu, dtype=torch.float32))
            self.sigma_raw = nn.Parameter(torch.tensor(radial_init_sigma, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown inhibition mode: {mode}")

        if device is not None:
            self.to(device)

    def forward(self, chi: torch.Tensor) -> torch.Tensor:
        """
        Compute inhibitory signal for a 2D feature map.

        Args:
            chi: Tensor of shape [B, H, W]
        Returns:
            inhib: Tensor of shape [B, H, W]
        """
        # Convolution expects [B, 1, H, W]
        x = chi.unsqueeze(1)

        if self.mode == "fixed":
            kernel = self.mask * self.psi
        elif self.mode == "learnable_scalar":
            psi = F.softplus(self.psi_raw)
            kernel = self.mask * psi
        elif self.mode == "learnable_kernel":
            kernel = F.softplus(self.kernel_raw) * self.mask
        elif self.mode == "radial":
            alpha = F.softplus(self.alpha_raw)
            mu = F.softplus(self.mu_raw) + 1e-6
            sigma = F.softplus(self.sigma_raw) + 1e-6
            kernel = alpha * torch.exp(-0.5 * ((self.dist - mu) / sigma) ** 2)
            kernel = kernel * self.mask
        else:
            raise RuntimeError("Invalid inhibition mode at forward.")

        kernel = kernel.to(x.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,k,k]

        # If feature map is smaller than kernel, skip inhibition.
        _, _, h, w = x.shape
        k = kernel.shape[-1]
        if h < k or w < k:
            return torch.zeros_like(chi)

        inhib = F.conv2d(x, kernel, padding=self.h)
        return inhib.squeeze(1)
