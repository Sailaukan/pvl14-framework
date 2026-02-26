from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from torch import Tensor

from ..utils import pad_like


class MDDMInferMixin:
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

    def _normalize_fix_mask(self, xt: Tensor, fix_mask: Optional[Tensor]) -> Tensor:
        if fix_mask is None:
            return torch.zeros_like(xt, dtype=torch.bool)
        normalized = fix_mask.bool()
        if normalized.shape != xt.shape:
            raise ValueError(f"fix_mask shape {normalized.shape} must match xt shape {xt.shape}")
        return normalized

    def _stochastic_prediction(
        self,
        logits: Tensor,
        logit_temperature: float,
        confidence_temperature: float,
        randomness: float,
        ratio: float,
    ):
        logits = logits.clone()
        logits[..., self.mask_index] += -1000000.0
        denom = max(logit_temperature, 1e-8)
        scaled_logits = logits / denom
        probs = torch.softmax(scaled_logits, dim=-1)
        sampled_logits = scaled_logits
        if confidence_temperature > 0.0 and randomness > 0.0:
            gumbel = -torch.log(
                -torch.log(
                    torch.rand(logits.shape, device=logits.device, generator=self.rng_generator) + 1e-8
                )
                + 1e-8
            )
            sampled_logits = sampled_logits + gumbel * confidence_temperature * randomness * max(0.0, 1.0 - ratio)
        preds = sampled_logits.argmax(dim=-1)
        pred_conf = probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)
        return preds, pred_conf, probs

    def step_threshold_regen(
        self,
        logits: Tensor,
        xt: Tensor,
        curr_step: int,
        num_steps: int,
        logit_temperature: float = 1.0,
        randomness: float = 1.0,
        confidence_temperature: float = 1.0,
        confidence_threshold: Optional[float] = None,
        min_conf_gain: Optional[float] = None,
        max_remask_frac: Optional[float] = None,
        allow_remask_unmasked: Optional[bool] = None,
        fix_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if xt.ndim > 3:
            raise NotImplementedError(
                "step_threshold_regen is implemented for Batch x Sequence x State Space shaped tensors."
            )
        if curr_step < 0 or num_steps < 1:
            raise ValueError("Invalid input values for curr_step or num_steps.")

        threshold = self.confidence_threshold if confidence_threshold is None else float(confidence_threshold)
        conf_gain = self.min_conf_gain if min_conf_gain is None else float(min_conf_gain)
        remask_frac = self.max_remask_frac if max_remask_frac is None else float(max_remask_frac)
        allow_remask = self.allow_remask_unmasked if allow_remask_unmasked is None else bool(allow_remask_unmasked)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0.0, 1.0]")
        if conf_gain < 0.0:
            raise ValueError("min_conf_gain must be >= 0.0")
        if not 0.0 <= remask_frac <= 1.0:
            raise ValueError("max_remask_frac must be in [0.0, 1.0]")

        xt = xt.clone()
        fix_mask = self._normalize_fix_mask(xt, fix_mask)
        ratio = 1.0 if num_steps <= 1 else max(0.0, min(1.0, curr_step / (num_steps - 1)))
        last_mask = xt == self.mask_index
        editable = ~fix_mask

        preds, pred_conf, probs = self._stochastic_prediction(
            logits=logits,
            logit_temperature=logit_temperature,
            confidence_temperature=confidence_temperature,
            randomness=randomness,
            ratio=ratio,
        )
        current_conf = probs.gather(-1, xt.unsqueeze(-1)).squeeze(-1)

        fill_mask = last_mask & editable
        xt[fill_mask] = preds[fill_mask]

        if curr_step == num_steps - 1:
            if allow_remask:
                replace_mask = editable & (~last_mask) & (current_conf < threshold) & (
                    pred_conf > current_conf + conf_gain
                )
                xt[replace_mask] = preds[replace_mask]
            return xt

        if not allow_remask:
            return xt

        remask_candidates = editable & (~last_mask) & (current_conf < threshold) & (
            pred_conf > current_conf + conf_gain
        )
        if not remask_candidates.any():
            return xt

        remask_budget = torch.ceil(editable.sum(dim=-1).float() * remask_frac * (1.0 - ratio)).long()
        remask_budget = torch.clamp(remask_budget, min=0, max=xt.shape[-1])

        candidate_scores = torch.where(remask_candidates, current_conf, torch.full_like(current_conf, float("inf")))
        remask_mask = torch.zeros_like(remask_candidates)
        for b in range(xt.shape[0]):
            k = min(int(remask_budget[b].item()), int(remask_candidates[b].sum().item()))
            if k <= 0:
                continue
            idx = torch.topk(candidate_scores[b], k=k, largest=False).indices
            remask_mask[b, idx] = True

        xt[remask_mask] = self.mask_index
        return xt

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
        confidence_threshold: Optional[float] = None,
        min_conf_gain: Optional[float] = None,
        max_remask_frac: Optional[float] = None,
        allow_remask_unmasked: Optional[bool] = None,
        fix_mask: Optional[Tensor] = None,
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
                fix_mask=fix_mask,
            )
        if self.decode_strategy == "threshold_regen":
            return self.step_threshold_regen(
                logits=logits,
                xt=xt,
                curr_step=curr_step,
                num_steps=num_steps,
                logit_temperature=logit_temperature,
                randomness=randomness,
                confidence_temperature=confidence_temperature,
                confidence_threshold=confidence_threshold,
                min_conf_gain=min_conf_gain,
                max_remask_frac=max_remask_frac,
                allow_remask_unmasked=allow_remask_unmasked,
                fix_mask=fix_mask,
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
