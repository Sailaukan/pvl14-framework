from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from torch import Tensor

from .distributions import DiscreteMaskedPrior, UniformTimeDistribution
from .noise import LogLinearExpNoiseTransform
from .utils import pad_like


class MDLM:
    def __init__(
        self,
        time_distribution: UniformTimeDistribution,
        prior_distribution: DiscreteMaskedPrior,
        noise_schedule: LogLinearExpNoiseTransform,
        device: str = "cpu",
        rng_generator: Optional[torch.Generator] = None,
        decode_strategy: Literal["self_path_planning"] = "self_path_planning",
    ):
        self.time_distribution = time_distribution
        self.prior_distribution = prior_distribution
        self.noise_schedule = noise_schedule
        self.device = device
        self.rng_generator = rng_generator
        self.decode_strategy = decode_strategy

        self.num_classes = prior_distribution.num_classes
        self.mask_index = int(prior_distribution.mask_dim)

    def to_device(self, device: str):
        self.device = device
        return self

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

    def get_num_steps_confidence(self, xt: Tensor, num_tokens_unmask: int = 1):
        nsteps = (xt == self.mask_index).sum(-1).max().item()
        if num_tokens_unmask == 1:
            return int(nsteps)
        return int(max(math.ceil(nsteps // num_tokens_unmask), 1))

    def stochastic_sample_from_categorical(
        self, logits: Tensor, temperature: float = 1.0, noise_scale: float = 1.0
    ):
        if temperature > 0:
            gumbel = -torch.log(
                -torch.log(
                    torch.rand(
                        logits.shape, device=logits.device, generator=self.rng_generator
                    )
                    + 1e-8
                )
                + 1e-8
            )
            logits = logits / temperature + noise_scale * gumbel
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
        return tokens, scores

    def topk_lowest_masking(self, scores: Tensor, cutoff_len: Tensor):
        sorted_scores, _ = scores.sort(dim=-1)
        cutoff_len = cutoff_len.clamp(min=0, max=scores.shape[-1] - 1)
        threshold = sorted_scores.gather(dim=-1, index=cutoff_len)
        return scores < threshold

    def step_self_path_planning(
        self,
        logits: Tensor,
        xt: Tensor,
        t: Tensor,
        curr_step: int,
        num_steps: int,
        logit_temperature: float = 1.0,
        randomness: float = 1.0,
        confidence_temperature: float = 1.0,
        score_type: Literal["confidence", "random"] = "confidence",
        fix_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if xt.ndim > 3:
            raise NotImplementedError(
                "step_self_path_planning is implemented for B x N or B x N x C shapes"
            )
        if curr_step < 0 or num_steps < 1:
            raise ValueError("Invalid input values for curr_step or num_steps")

        xt = xt.clone()
        if fix_mask is None:
            fix_mask = torch.zeros_like(xt, dtype=torch.bool)
        else:
            fix_mask = fix_mask.bool()

        last_mask = xt == self.mask_index
        unmask_candidates = last_mask & ~fix_mask

        x1_pred, logp = self.stochastic_sample_from_categorical(
            logits,
            temperature=logit_temperature,
            noise_scale=confidence_temperature,
        )

        if curr_step == num_steps - 1:
            xt[last_mask] = x1_pred[last_mask]
            return xt

        if score_type == "confidence":
            score = logp
        else:
            score = torch.rand_like(logp).clamp_min(1e-8).log()

        score = score.masked_fill(fix_mask, float("inf"))
        score = score.clone()
        score[unmask_candidates] *= randomness

        editable_count = (~fix_mask).sum(dim=1, keepdim=True)
        num_to_mask = torch.clamp(
            (editable_count.float() * t.unsqueeze(-1)).long(), max=xt.shape[-1] - 1
        )

        mask = self.topk_lowest_masking(score, num_to_mask)
        xt[mask] = self.mask_index

        mask_to_x1 = last_mask & ~mask
        xt[mask_to_x1] = x1_pred[mask_to_x1]
        return xt

    def step_confidence(
        self,
        logits: Tensor,
        xt: Tensor,
        curr_step: int,
        num_steps: int,
        logit_temperature: float = 1.0,
        randomness: float = 1.0,
        confidence_temperature: float = 1.0,
        num_tokens_unmask: int = 1,
        score_type: Literal["confidence", "random"] = "confidence",
        fix_mask: Optional[Tensor] = None,
    ) -> Tensor:
        del num_tokens_unmask
        if self.decode_strategy != "self_path_planning":
            raise ValueError(f"Unsupported decode_strategy={self.decode_strategy}")

        if num_steps <= 1:
            t_scalar = 0.0
        else:
            ratio = curr_step / max(num_steps - 1, 1)
            t_scalar = max(0.0, min(1.0, 1.0 - ratio))
        t = torch.full((xt.shape[0],), t_scalar, device=xt.device)

        return self.step_self_path_planning(
            logits=logits,
            xt=xt,
            t=t,
            curr_step=curr_step,
            num_steps=num_steps,
            logit_temperature=logit_temperature,
            randomness=randomness,
            confidence_temperature=confidence_temperature,
            score_type=score_type,
            fix_mask=fix_mask,
        )

    def step_argmax(self, model_out: Tensor):
        return model_out.argmax(dim=-1)
