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
        decode_strategy: Literal["confidence", "self_path_planning"] = "confidence",
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

    def step(
        self,
        logits: Tensor,
        t: Tensor,
        xt: Tensor,
        dt: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        sigma_t = self.noise_schedule.calculate_sigma(t, logits.device)
        sigma_s = self.noise_schedule.calculate_sigma(t - dt, logits.device)
        alpha_t = torch.exp(-sigma_t)
        alpha_s = torch.exp(-sigma_s)
        p_mask_s = 1 - alpha_s
        alpha_t = pad_like(alpha_t, logits)
        alpha_s = pad_like(alpha_s, logits)
        p_mask_s = pad_like(p_mask_s, logits)

        log_p_x0 = self._subs_parameterization(logits, xt) / temperature
        if p_mask_s.ndim != log_p_x0.ndim:
            raise ValueError(f"Dimension Mistmatch {p_mask_s.shape} {log_p_x0.shape}")

        prob_s_given_t = log_p_x0.exp() * (alpha_s - alpha_t)
        prob_s_given_t[..., self.mask_index] = p_mask_s[..., 0]
        sampled_x = self._sample_categorical(prob_s_given_t)
        carry_over_unmask = (xt != self.mask_index).to(xt.dtype)
        return carry_over_unmask * xt + (1 - carry_over_unmask) * sampled_x

    def _sample_categorical(self, categorical_probs: Tensor) -> Tensor:
        gumbel_norm = (
            1e-10
            - (
                torch.rand(
                    *categorical_probs.shape,
                    device=categorical_probs.device,
                    generator=self.rng_generator,
                )
                + 1e-10
            ).log()
        )
        scaled_probability = categorical_probs / gumbel_norm
        return scaled_probability.argmax(dim=-1)

    def get_num_steps_confidence(self, xt: Tensor, num_tokens_unmask: int = 1):
        nsteps = (xt == self.mask_index).sum(-1).max().item()
        if num_tokens_unmask == 1:
            return int(nsteps)
        return int(max(math.ceil(nsteps // num_tokens_unmask), 1))

    def step_auto_regressive(
        self,
        logits: Tensor,
        xt: Tensor,
        logit_temperature: float = 1.0,
    ):
        xt = xt.clone()
        log_p_x0 = self._subs_parameterization(logits, xt)
        probs = torch.softmax(log_p_x0 / logit_temperature, dim=-1)
        preds = torch.distributions.Categorical(probs=probs).sample()

        mask = xt == self.mask_index
        next_idx = torch.where(mask)[1][0]
        to_replace = torch.zeros_like(xt)
        to_replace[:, next_idx] = 1
        to_replace = (mask.float() * to_replace.float()).bool()

        xt[to_replace] = preds[to_replace]
        return xt

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
                "step_confidence is implemented for Batch x Sequence x State Space shaped tensors."
            )
        if curr_step < 0 or num_steps < 1:
            raise ValueError("Invalid input values for curr_step, num_steps.")
        xt = xt.clone()
        if fix_mask is None:
            fix_mask = torch.zeros_like(xt).bool()
        last_mask = xt == self.mask_index
        unmask_candidates = last_mask & ~fix_mask
        x1_pred, logp = self.stochastic_sample_from_categorical(
            logits, temperature=logit_temperature, noise_scale=confidence_temperature
        )
        if curr_step == num_steps - 1:
            xt[last_mask] = x1_pred[last_mask]
        else:
            if score_type == "confidence":
                score = logp
            elif score_type == "random":
                score = torch.rand_like(logp).log()
            else:
                raise ValueError(f"Unknown score_type={score_type}")
            score = score.masked_fill(fix_mask.squeeze(-1), float("inf"))
            score[unmask_candidates.squeeze(-1)] *= randomness
            num_to_mask = torch.clamp(
                ((~fix_mask).sum(dim=1, keepdim=True).float() * t.unsqueeze(-1)).long(),
                max=xt.shape[-1] - 1,
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
    ) -> Tensor:
        if self.decode_strategy == "self_path_planning":
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
                score_type="confidence",
                fix_mask=None,
            )

        if xt.ndim > 3:
            raise NotImplementedError(
                "step_confidence is implemented for Batch x Sequence x State Space shaped tensors."
            )
        if curr_step < 0 or num_steps < 1 or num_tokens_unmask < 1:
            raise ValueError("Invalid input values for curr_step, num_steps, or num_tokens_unmask.")

        xt = xt.clone()
        log_p_x0 = self._subs_parameterization(logits, xt)
        probs = torch.softmax(log_p_x0 / logit_temperature, dim=-1)
        preds = torch.distributions.Categorical(probs=probs).sample()

        confidence = probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)
        ratio = curr_step / (num_steps - 1)
        gumbel_sample = -torch.log(
            -torch.log(torch.rand(xt.shape, device=logits.device, generator=self.rng_generator))
        )
        gumbel_noise = gumbel_sample * randomness * (1 - ratio)
        confidence = (torch.log(confidence) + gumbel_noise) / confidence_temperature

        mask = xt == self.mask_index
        confidence[~mask] = -torch.inf

        _, idx_mask = torch.topk(confidence, k=num_tokens_unmask, dim=-1)
        to_replace = torch.zeros_like(confidence)
        to_replace.scatter_(1, idx_mask, 1)
        to_replace = to_replace.bool() & mask.bool()
        xt[to_replace] = preds[to_replace]
        return xt

    def step_confidence_margin(
        self,
        logits: Tensor,
        xt: Tensor,
        curr_step: int,
        num_steps: int,
        logit_temperature: float = 1.0,
        randomness: float = 1.0,
        confidence_temperature: float = 1.0,
        num_tokens_unmask: int = 1,
    ) -> Tensor:
        if xt.ndim > 3:
            raise NotImplementedError(
                "step_confidence is implemented for Batch x Sequence x State Space shaped tensors."
            )
        if curr_step < 0 or num_steps < 1 or num_tokens_unmask < 1:
            raise ValueError("Invalid input values for curr_step, num_steps, or num_tokens_unmask.")

        xt = xt.clone()
        log_p_x0 = self._subs_parameterization(logits, xt)
        probs = torch.softmax(log_p_x0 / logit_temperature, dim=-1)
        preds = torch.stack([torch.multinomial(prob, num_samples=2, replacement=False) for prob in probs])
        confidence_first = probs.gather(-1, preds[:, :, 0].unsqueeze(-1)).squeeze(-1)
        confidence_second = probs.gather(-1, preds[:, :, 1].unsqueeze(-1)).squeeze(-1)
        confidence = confidence_first - confidence_second
        preds = preds[:, :, 0]

        ratio = curr_step / (num_steps - 1)
        gumbel_sample = -torch.log(
            -torch.log(torch.rand(xt.shape, device=logits.device, generator=self.rng_generator))
        )
        gumbel_noise = gumbel_sample * randomness * (1 - ratio)
        confidence = (torch.log(confidence) + gumbel_noise) / confidence_temperature

        mask = xt == self.mask_index
        confidence[~mask] = -torch.inf

        _, idx_mask = torch.topk(confidence, k=num_tokens_unmask, dim=-1)
        to_replace = torch.zeros_like(confidence)
        to_replace.scatter_(1, idx_mask, 1)
        to_replace = to_replace.bool() & mask.bool()
        xt[to_replace] = preds[to_replace]
        return xt

    def step_argmax(self, model_out: Tensor):
        return model_out.argmax(dim=-1)

    def calculate_score(self, logits: Tensor, x: Tensor, t: Tensor):
        sigma_t = self.noise_schedule.calculate_sigma(t, logits.device)
        log_ratio = -torch.log(torch.expm1(sigma_t))

        masked_log_score = logits + pad_like(log_ratio, logits)
        masked_log_score[..., self.mask_index] = 0

        unmasked_log_score = torch.full_like(logits, -1000000.0)
        unmasked_log_score.scatter_(-1, x[..., None], 0)
        unmasked_log_score[..., self.mask_index] = -pad_like(log_ratio, logits[..., 0])

        masked_indices = (x == self.mask_index).to(logits.dtype)[..., None]
        log_score = masked_log_score * masked_indices + unmasked_log_score * (1 - masked_indices)
        return log_score.exp()
