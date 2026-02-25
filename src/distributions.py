from __future__ import annotations

import math
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
    rng_generator: Optional[torch.Generator] = None

    def __post_init__(self):
        if self.discrete_time:
            self.min_t = 0.0
            self.max_t = 1.0
            if self.nsteps is None:
                raise ValueError("nsteps must not be None and must be specified for discrete time")
        if not 0 <= self.min_t < 1.0:
            raise ValueError("min_t must be greater than or equal to 0 and less than 1.0")
        if not 0 < self.max_t <= 1.0:
            raise ValueError("max_t must be greater than 0 and less than or equal to 1.0")
        if self.min_t >= self.max_t:
            raise ValueError("min_t must be less than max_t")

    def sample(
        self,
        n_samples: Union[int, Tuple[int, ...], torch.Size],
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        if rng_generator is None:
            rng_generator = self.rng_generator
        if self.discrete_time:
            if self.nsteps is None:
                raise ValueError("nsteps cannot be None for discrete time sampling")
            shape = (n_samples,) if isinstance(n_samples, int) else tuple(n_samples)
            return torch.randint(0, self.nsteps, shape, device=device, generator=rng_generator)

        time_step = torch.rand(n_samples, device=device, generator=rng_generator)
        if self.min_t and self.max_t and self.min_t > 0:
            time_step = time_step * (self.max_t - self.min_t) + self.min_t
        return time_step


@dataclass
class AntitheticUniformTimeDistribution(UniformTimeDistribution):
    sampling_eps: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.sampling_eps < 1.0:
            raise ValueError("sampling_eps must be in [0.0, 1.0)")

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

        time_step = torch.rand((n_total,), device=device, generator=rng_generator)
        offset = torch.arange(n_total, device=device) / n_total
        time_step = (time_step / n_total + offset) % 1

        if self.discrete_time:
            if self.nsteps is None:
                raise ValueError("nsteps cannot be None for discrete time sampling")
            return (time_step * self.nsteps).long().view(shape)

        time_step = (1 - self.sampling_eps) * time_step + self.sampling_eps
        if self.min_t and self.max_t and self.min_t > 0:
            time_step = time_step * (self.max_t - self.min_t) + self.min_t
        return time_step.view(shape)


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
            self.prior_dist = torch.zeros((self.num_classes))
            self.prior_dist[-1] = 1.0
        else:
            if self.mask_dim is None:
                self.mask_dim = self.num_classes
            self.num_classes = self.num_classes + 1
            self.prior_dist = torch.zeros((self.num_classes))
            self.prior_dist[-1] = 1.0
        if torch.sum(self.prior_dist).item() - 1.0 >= 1e-5:
            raise ValueError("Invalid probability distribution. Must sum to 1.0")

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

    def pad_sample(self, sample: Tensor) -> Tensor:
        zeros = torch.zeros((*sample.shape[:-1], 1), dtype=torch.float, device=sample.device)
        padded_sample = torch.cat((sample, zeros), dim=-1)
        return padded_sample
