from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, Union, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class TimeDistribution(Protocol):
    def sample(
        self,
        n_samples: Union[int, Tuple[int, ...], torch.Size],
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        ...


def _as_shape(n_samples: Union[int, Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
    return (n_samples,) if isinstance(n_samples, int) else tuple(n_samples)


def _validate_time_bounds(min_t: float, max_t: float):
    if not 0.0 <= min_t < 1.0:
        raise ValueError("min_t must be in [0.0, 1.0)")
    if not 0.0 < max_t <= 1.0:
        raise ValueError("max_t must be in (0.0, 1.0]")
    if min_t >= max_t:
        raise ValueError("min_t must be strictly less than max_t")


@dataclass
class DiscreteUniformTD:
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

        shape = _as_shape(n_samples)
        return torch.randint(
            0, self.nsteps, shape, device=device, generator=rng_generator
        )


@dataclass
class DiscreteAntitheticUniformTD(DiscreteUniformTD):
    def sample(
        self,
        n_samples: Union[int, Tuple[int, ...], torch.Size],
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        if rng_generator is None:
            rng_generator = self.rng_generator

        shape = _as_shape(n_samples)
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
class DiscreteSymmetricUniformTD(DiscreteUniformTD):
    def sample(
        self,
        n_samples: Union[int, Tuple[int, ...], torch.Size],
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        if rng_generator is None:
            rng_generator = self.rng_generator

        if not isinstance(n_samples, int):
            n_samples = n_samples[0]

        time_step = torch.randint(
            0,
            self.nsteps,
            size=(n_samples // 2 + 1,),
            device=device,
            generator=rng_generator,
        )
        return torch.cat([time_step, self.nsteps - time_step - 1], dim=0)[:n_samples]


@dataclass
class ContinuousUniformTD:
    min_t: float = 0.0
    max_t: float = 1.0
    sampling_eps: float = 0.0
    rng_generator: Optional[torch.Generator] = None

    def __post_init__(self):
        _validate_time_bounds(self.min_t, self.max_t)
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

        shape = _as_shape(n_samples)
        time_step = torch.rand(shape, device=device, generator=rng_generator)
        if self.sampling_eps > 0:
            time_step = (1 - self.sampling_eps) * time_step + self.sampling_eps
        return time_step * (self.max_t - self.min_t) + self.min_t


@dataclass
class ContinuousAntitheticUniformTD(ContinuousUniformTD):
    def sample(
        self,
        n_samples: Union[int, Tuple[int, ...], torch.Size],
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        if rng_generator is None:
            rng_generator = self.rng_generator

        shape = _as_shape(n_samples)
        if len(shape) == 0:
            raise ValueError("n_samples must describe at least one dimension")
        n_total = math.prod(shape)
        if n_total <= 0:
            raise ValueError("n_samples must be > 0")

        time_step = torch.rand((n_total,), device=device, generator=rng_generator)
        offset = torch.arange(n_total, device=device) / n_total
        time_step = (time_step / n_total + offset) % 1
        if self.sampling_eps > 0:
            time_step = (1 - self.sampling_eps) * time_step + self.sampling_eps
        time_step = time_step * (self.max_t - self.min_t) + self.min_t
        return time_step.view(shape)


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
