from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..utils import pad_like


class MDDMTrainMixin:
    def sample_prior(self, *args, **kwargs) -> Tensor:
        if "device" not in kwargs:
            kwargs["device"] = self.device
        kwargs["rng_generator"] = self.rng_generator
        return self.prior_distribution.sample(*args, **kwargs)

    def sample_time(self, *args, **kwargs) -> Tensor:
        if "device" not in kwargs:
            kwargs["device"] = self.device
        kwargs["rng_generator"] = self.rng_generator
        return self.time_distribution.sample(*args, **kwargs)

    def interpolate(self, data: Tensor, t: Tensor):
        if data.dtype == torch.float and data.ndim > 2:
            x0 = data.argmax(-1)
        else:
            x0 = data
        sigma = self.noise_schedule.calculate_sigma(t, data.device)
        alpha = self.noise_schedule.sigma_to_alpha(sigma)
        p_mask = 1 - alpha
        p_mask = pad_like(p_mask, x0)
        mask_indices = (
            torch.rand(*x0.shape, device=x0.device, generator=self.rng_generator)
            < p_mask
        )
        xt = torch.where(mask_indices, self.mask_index, x0)
        return xt

    def forward_process(self, data: Tensor, t: Tensor) -> Tensor:
        return self.interpolate(data, t)

    def _subs_parameterization(self, logits: Tensor, xt: Tensor) -> Tensor:
        logits = logits.clone()
        logits[..., self.mask_index] += -1000000.0
        logprobs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked_indices = xt != self.mask_index
        logprobs[unmasked_indices] = -1000000.0
        logprobs[unmasked_indices, xt[unmasked_indices]] = 0
        return logprobs

    def loss(
        self,
        logits: Tensor,
        target: Tensor,
        xt: Tensor,
        time: Tensor,
        mask: Optional[Tensor] = None,
        use_weight: bool = True,
        global_mean: bool = False,
    ):
        logprobs = self._subs_parameterization(logits, xt)
        log_p_theta = torch.gather(
            input=logprobs, dim=-1, index=target[..., None]
        ).squeeze(-1)

        sigma = self.noise_schedule.calculate_sigma(time, target.device)
        dsigma = self.noise_schedule.d_dt_sigma(time, target.device)
        loss = -log_p_theta
        if use_weight:
            loss = loss * (dsigma / torch.expm1(sigma))[:, None]

        if global_mean:
            if mask is not None:
                loss = loss * mask
                loss = loss.sum() / mask.sum().clamp_min(1)
            else:
                loss = loss.sum() / logits.size(1)
        else:
            if mask is not None:
                loss = loss * mask
                num_non_masked_elements = torch.sum(mask, dim=-1).clamp_min(1)
                loss = torch.sum(loss, dim=-1) / num_non_masked_elements
            else:
                loss = torch.sum(loss, dim=-1) / logits.size(1)
        return loss
