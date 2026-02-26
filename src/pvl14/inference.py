from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

import torch
from torch import Tensor


def _validate_time_bounds(min_t: float, max_t: float):
    if not 0.0 <= min_t < 1.0:
        raise ValueError("min_t must be in [0.0, 1.0)")
    if not 0.0 < max_t <= 1.0:
        raise ValueError("max_t must be in (0.0, 1.0]")
    if min_t >= max_t:
        raise ValueError("min_t must be strictly less than max_t")


@dataclass
class _TimeSchedule:
    nsteps: int
    min_t: float = 0.0
    max_t: float = 1.0
    device: Union[str, torch.device] = "cpu"

    def __post_init__(self):
        if self.nsteps < 2:
            raise ValueError("nsteps must be >= 2")
        _validate_time_bounds(self.min_t, self.max_t)

    def _curve(self, s: Tensor) -> Tensor:
        raise NotImplementedError

    def generate(
        self,
        nsteps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tensor:
        n = nsteps if nsteps is not None else self.nsteps
        d = device if device is not None else self.device
        if n < 2:
            raise ValueError("nsteps must be >= 2")
        s = torch.linspace(0.0, 1.0, n, device=d)
        return self.min_t + (self.max_t - self.min_t) * self._curve(s)


class LinearTimeSchedule(_TimeSchedule):
    def _curve(self, s: Tensor) -> Tensor:
        return 1.0 - s


class CosineTimeSchedule(_TimeSchedule):
    def _curve(self, s: Tensor) -> Tensor:
        return torch.cos(s * torch.pi / 2).clamp(0.0, 1.0)


@dataclass
class ExponentialTimeSchedule(_TimeSchedule):
    k: float = 5.0

    def __post_init__(self):
        super().__post_init__()
        if self.k <= 0.0:
            raise ValueError("k must be > 0.0")

    def _curve(self, s: Tensor) -> Tensor:
        exp_neg_k = torch.exp(torch.tensor(-self.k, device=s.device, dtype=s.dtype))
        return ((torch.exp(-self.k * s) - exp_neg_k) / (1.0 - exp_neg_k)).clamp(0.0, 1.0)


def get_time_deltas(schedule: Tensor) -> Tensor:
    if schedule.ndim != 1:
        raise ValueError("schedule must be a 1D tensor")
    if schedule.numel() < 2:
        raise ValueError("schedule must contain at least 2 values")
    deltas = schedule[:-1] - schedule[1:]
    if torch.any(deltas <= 0):
        raise ValueError("schedule must be strictly decreasing")
    return deltas


def run_inference_loop(
    mddm,
    model_fn: Callable[[Tensor, Tensor], Tensor],
    x_init: Tensor,
    schedule: Union[Tensor, LinearTimeSchedule, CosineTimeSchedule, ExponentialTimeSchedule],
    strategy: Literal["step", "confidence"] = "step",
    temperature: float = 1.0,
    logit_temperature: float = 1.0,
    randomness: float = 1.0,
    confidence_temperature: float = 1.0,
    num_tokens_unmask: int = 1,
    confidence_threshold: Optional[float] = None,
    min_conf_gain: Optional[float] = None,
    max_remask_frac: Optional[float] = None,
    allow_remask_unmasked: Optional[bool] = None,
    fix_mask: Optional[Tensor] = None,
) -> Tensor:
    if strategy not in ("step", "confidence"):
        raise ValueError(f"Unknown strategy: {strategy!r}. Must be 'step' or 'confidence'.")

    if isinstance(schedule, Tensor):
        schedule_tensor = schedule
    else:
        schedule_tensor = schedule.generate(device=x_init.device)

    schedule_tensor = schedule_tensor.to(device=x_init.device, dtype=torch.float32)
    deltas = get_time_deltas(schedule_tensor)
    total_steps = deltas.numel()
    num_steps_for_confidence = max(total_steps, 2)
    batch_size = x_init.shape[0]

    x = x_init
    for i in range(total_steps):
        t = schedule_tensor[i : i + 1].expand(batch_size)
        logits = model_fn(x, t)

        if strategy == "step":
            dt = deltas[i : i + 1].expand(batch_size)
            x = mddm.step(logits=logits, t=t, xt=x, dt=dt, temperature=temperature)
        else:
            x = mddm.step_confidence(
                logits=logits,
                xt=x,
                curr_step=i,
                num_steps=num_steps_for_confidence,
                logit_temperature=logit_temperature,
                randomness=randomness,
                confidence_temperature=confidence_temperature,
                num_tokens_unmask=num_tokens_unmask,
                confidence_threshold=confidence_threshold,
                min_conf_gain=min_conf_gain,
                max_remask_frac=max_remask_frac,
                allow_remask_unmasked=allow_remask_unmasked,
                fix_mask=fix_mask,
            )

    return x
