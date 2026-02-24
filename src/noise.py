from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor


@dataclass
class LogLinearExpNoiseTransform:
    eps: float = 1.0e-3

    def calculate_sigma(self, t: Tensor, device: Union[str, torch.device] = "cpu") -> Tensor:
        if t.max() > 1:
            raise ValueError(f"Invalid value: max continuous time is 1, but got {t.max().item()}")
        return -torch.log1p(-(1 - self.eps) * t).to(device)

    def d_dt_sigma(self, t: Tensor, device: Union[str, torch.device] = "cpu") -> Tensor:
        derivative = (1 - self.eps) / (1 - (1 - self.eps) * t)
        return derivative.to(device)

    def sigma_to_alpha(self, sigma: Tensor) -> Tensor:
        return torch.exp(-sigma)
