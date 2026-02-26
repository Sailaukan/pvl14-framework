from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor


@dataclass
class LogLinearExpNoiseTransform:
    eps: float = 1.0e-3

    def calculate_sigma(
        self, t: Tensor, device: Union[str, torch.device] = "cpu"
    ) -> Tensor:
        if t.max() > 1:
            raise ValueError(
                f"Invalid value: max continuous time is 1, but got {t.max().item()}"
            )
        return -torch.log1p(-(1 - self.eps) * t).to(device)

    def d_dt_sigma(self, t: Tensor, device: Union[str, torch.device] = "cpu") -> Tensor:
        derivative = (1 - self.eps) / (1 - (1 - self.eps) * t)
        return derivative.to(device)

    def sigma_to_alpha(self, sigma: Tensor) -> Tensor:
        return torch.exp(-sigma)


@dataclass
class CosineNoiseTransform:
    """Cosine noise schedule: masks tokens slowly at start/end, faster in the middle."""

    eps: float = 1.0e-3

    def calculate_sigma(
        self, t: Tensor, device: Union[str, torch.device] = "cpu"
    ) -> Tensor:
        if t.max() > 1:
            raise ValueError(
                f"Invalid value: max continuous time is 1, but got {t.max().item()}"
            )
        alpha = torch.cos(t * torch.pi / 2).clamp(min=self.eps)
        return -torch.log(alpha).to(device)

    def d_dt_sigma(self, t: Tensor, device: Union[str, torch.device] = "cpu") -> Tensor:
        return (torch.pi / 2 * torch.tan(t * torch.pi / 2)).to(device)

    def sigma_to_alpha(self, sigma: Tensor) -> Tensor:
        return torch.exp(-sigma)


@dataclass
class LinearNoiseTransform:
    """Linear noise schedule: masks tokens at a constant rate."""

    eps: float = 1.0e-3

    def calculate_sigma(
        self, t: Tensor, device: Union[str, torch.device] = "cpu"
    ) -> Tensor:
        if t.max() > 1:
            raise ValueError(
                f"Invalid value: max continuous time is 1, but got {t.max().item()}"
            )
        alpha = (1 - t).clamp(min=self.eps)
        return -torch.log(alpha).to(device)

    def d_dt_sigma(self, t: Tensor, device: Union[str, torch.device] = "cpu") -> Tensor:
        return (1 / (1 - t).clamp(min=self.eps)).to(device)

    def sigma_to_alpha(self, sigma: Tensor) -> Tensor:
        return torch.exp(-sigma)
