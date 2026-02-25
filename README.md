# PVL14

A lightweight open-source Python framework for masked discrete diffusion.

## What it is

`PVL14` provides a minimal MDLM-style implementation for token-based generative modeling with iterative mask/unmask decoding.

## Features

- Minimal `MDLM` class for masked discrete diffusion.
- Confidence-based iterative decoding.
- Re-masking and re-generation via self-path-planning style sampling.
- Small, readable codebase built on PyTorch.

## Installation

```bash
pip install -e .
```

## Quick start

```python
import torch
from pvl14_framework import (
    MDLM,
    UniformTD,
    AntitheticUniformTD,
    LogLinearExpNoiseTransform,
    DiscreteMaskedPrior,
)

prior = DiscreteMaskedPrior(num_classes=100, mask_dim=99)
mdlm = MDLM(
    time_distribution=UniformTD(),
    prior_distribution=prior,
    noise_schedule=LogLinearExpNoiseTransform(),
)

x = torch.full((2, 16), 99)              # masked tokens
logits = torch.randn(2, 16, 100)         # model output logits
x_next = mdlm.step_confidence(logits, x, curr_step=0, num_steps=8)

# Optional: antithetic time sampling (GenMol-style)
mdlm_antithetic = MDLM(
    time_distribution=AntitheticUniformTD(sampling_eps=1e-3),
    prior_distribution=prior,
    noise_schedule=LogLinearExpNoiseTransform(),
)
```

## Status

Early-stage and intentionally minimal.
# pvl14-framework
