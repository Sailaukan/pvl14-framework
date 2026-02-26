from __future__ import annotations

from typing import Literal, Optional

import torch

from ..distributions import DiscreteMaskedPrior, UniformTD
from .mddm_infer import MDDMInferMixin
from .mddm_train import MDDMTrainMixin
from ..noise import LogLinearExpNoiseTransform


class MDDM(MDDMTrainMixin, MDDMInferMixin):
    def __init__(
        self,
        time_distribution: UniformTD,
        prior_distribution: DiscreteMaskedPrior,
        noise_schedule: LogLinearExpNoiseTransform,
        device: str = "cpu",
        rng_generator: Optional[torch.Generator] = None,
        decode_strategy: Literal["confidence", "self_path_planning", "threshold_regen"] = "confidence",
        confidence_threshold: float = 0.50,
        min_conf_gain: float = 0.00,
        max_remask_frac: float = 0.25,
        allow_remask_unmasked: bool = True,
    ):
        self.time_distribution = time_distribution
        self.prior_distribution = prior_distribution
        self.noise_schedule = noise_schedule
        self.device = device
        self.rng_generator = rng_generator
        self.decode_strategy = decode_strategy
        self.confidence_threshold = float(confidence_threshold)
        self.min_conf_gain = float(min_conf_gain)
        self.max_remask_frac = float(max_remask_frac)
        self.allow_remask_unmasked = bool(allow_remask_unmasked)

        self.num_classes = prior_distribution.num_classes
        self.mask_index = int(prior_distribution.mask_dim)
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0.0, 1.0]")
        if self.min_conf_gain < 0.0:
            raise ValueError("min_conf_gain must be >= 0.0")
        if not 0.0 <= self.max_remask_frac <= 1.0:
            raise ValueError("max_remask_frac must be in [0.0, 1.0]")

    def to_device(self, device: str):
        self.device = device
        return self
