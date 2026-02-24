from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor


@dataclass
class UniformTimeDistribution:
    discrete_time: bool = False
    nsteps: Optional[int] = None
    min_t: float = 0.0
    max_t: float = 1.0

    def sample(
        self,
        n_samples: Union[int, Tuple[int, ...], torch.Size],
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        if self.discrete_time:
            if self.nsteps is None:
                raise ValueError("nsteps cannot be None for discrete time sampling")
            time_step = torch.randint(0, self.nsteps, (n_samples,), device=device, generator=rng_generator)
            return time_step

        time_step = torch.rand(n_samples, device=device, generator=rng_generator)
        return time_step * (self.max_t - self.min_t) + self.min_t


@dataclass
class DiscreteMaskedPrior:
    num_classes: int
    mask_dim: Optional[int] = None
    inclusive: bool = True

    def __post_init__(self):
        if self.inclusive:
            if self.mask_dim is None:
                self.mask_dim = self.num_classes - 1
            if self.mask_dim >= self.num_classes:
                raise ValueError("For inclusive=True, mask_dim must be < num_classes")
        else:
            if self.mask_dim is None:
                self.mask_dim = self.num_classes
            self.num_classes = self.num_classes + 1

    def sample(
        self,
        shape: Tuple,
        mask: Optional[Tensor] = None,
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        del rng_generator
        samples = torch.ones(shape, dtype=torch.int64, device=device) * int(self.mask_dim)
        if mask is not None:
            samples = samples * mask[(...,) + (None,) * (len(samples.shape) - len(mask.shape))]
        return samples

    def is_masked(self, sample: Tensor) -> Tensor:
        return (sample == int(self.mask_dim)).float()
