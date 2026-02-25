from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor


@dataclass
class UniformTD:
    nsteps: int
    rng_generator: Optional[torch.Generator] = None

    def __post_init__(self):
        if self.nsteps <= 0:
            raise ValueError("nsteps must be greater than 0")

    def sample(
        self,
        n_samples: Union[int, Tuple[int, ...], torch.Size],
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        if rng_generator is None:
            rng_generator = self.rng_generator

        shape = (n_samples,) if isinstance(n_samples, int) else tuple(n_samples)
        return torch.randint(
            0, self.nsteps, shape, device=device, generator=rng_generator
        )


@dataclass
class AntitheticUniformTD(UniformTD):
    def sample(
        self,
        n_samples: Union[int, Tuple[int, ...], torch.Size],
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        if rng_generator is None:
            rng_generator = self.rng_generator

        shape = (n_samples,) if isinstance(n_samples, int) else tuple(n_samples)
        if len(shape) == 0:
            raise ValueError("n_samples must describe at least one dimension")
        n_total = math.prod(shape)
        if n_total <= 0:
            raise ValueError("n_samples must be > 0")

        # Stratified sampling to ensure even distribution across the steps
        time_step = torch.rand((n_total,), device=device, generator=rng_generator)
        offset = torch.arange(n_total, device=device) / n_total
        time_step = (time_step / n_total + offset) % 1

        # Map the float range [0, 1) to discrete step indices [0, nsteps-1]
        return (time_step * self.nsteps).long().view(shape)


@dataclass
class DiscreteMaskedPrior:
    num_classes: int  # Vocabulary size (includes the mask token)
    mask_dim: Optional[int] = None

    def __post_init__(self):
        if self.num_classes <= 0:
            raise ValueError("num_classes must be > 0")
        if self.mask_dim is None:
            self.mask_dim = self.num_classes - 1  # Last index is the mask token
        if self.mask_dim >= self.num_classes:
            raise ValueError("mask_dim must be < num_classes")

    def sample(
        self,
        shape: Tuple,
        mask: Optional[Tensor] = None,
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[
            torch.Generator
        ] = None,  # Unused, kept for API compatibility
    ) -> Tensor:
        samples = torch.ones(shape, dtype=torch.int64, device=device) * int(
            self.mask_dim
        )
        if mask is not None:
            samples = (
                samples
                * mask[(...,) + (None,) * (len(samples.shape) - len(mask.shape))]
            )
        return samples

    def is_masked(self, sample: Tensor) -> Tensor:
        return (sample == int(self.mask_dim)).float()
