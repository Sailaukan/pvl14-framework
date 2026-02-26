![PVL14 Framework Cover](pvl14_cover.png)

# PVL14

Lightweight PyTorch code for masked discrete diffusion with MDLM-style decoding.

## Current codebase

Core classes exported by `src/__init__.py`:

- `MDLM`
- `UniformTD`
- `AntitheticUniformTD`
- `SymmetricUniformTD`
- `DiscreteMaskedPrior`
- `LogLinearExpNoiseTransform`
- `LinearTimeSchedule`
- `CosineTimeSchedule`
- `run_inference_loop`

Additional schedules available from `src/noise.py`:

- `CosineNoiseTransform`
- `LinearNoiseTransform`

## Installation

```bash
pip install -e .
```

## Quick start (matches current APIs)

```python
import torch
from src import (
    MDLM,
    UniformTD,
    DiscreteMaskedPrior,
    LogLinearExpNoiseTransform,
)

batch, seq_len, vocab = 2, 16, 100
mask_idx = vocab - 1

prior = DiscreteMaskedPrior(num_classes=vocab, mask_dim=mask_idx)
mdlm = MDLM(
    time_distribution=UniformTD(nsteps=8),
    prior_distribution=prior,
    noise_schedule=LogLinearExpNoiseTransform(),
)

# Start from a fully masked sample
xt = prior.sample(shape=(batch, seq_len))
logits = torch.randn(batch, seq_len, vocab)

xt_next = mdlm.step_confidence(
    logits=logits,
    xt=xt,
    curr_step=0,
    num_steps=8,
    num_tokens_unmask=2,
)
```

## Forward noising and loss

```python
import torch
from src import MDLM, UniformTD, DiscreteMaskedPrior, LogLinearExpNoiseTransform

batch, seq_len, vocab = 4, 12, 64
prior = DiscreteMaskedPrior(num_classes=vocab)
mdlm = MDLM(
    time_distribution=UniformTD(nsteps=16),
    prior_distribution=prior,
    noise_schedule=LogLinearExpNoiseTransform(),
)

x0 = torch.randint(0, vocab - 1, (batch, seq_len))
t = torch.rand(batch)  # continuous time in [0, 1]
xt = mdlm.forward_process(x0, t)
logits = torch.randn(batch, seq_len, vocab)

loss_per_sample = mdlm.loss(logits=logits, target=x0, xt=xt, time=t)
```

## Decoding options

- `decode_strategy="confidence"` (default): unmask top-confidence positions each step.
- `decode_strategy="self_path_planning"`: uses re-masking / regeneration behavior through `step_confidence(...)`.

## Inference schedules + helper loop

```python
import torch
from src import (
    MDLM,
    UniformTD,
    DiscreteMaskedPrior,
    LogLinearExpNoiseTransform,
    LinearTimeSchedule,
    CosineTimeSchedule,
    run_inference_loop,
)

batch, seq_len, vocab = 2, 16, 100
prior = DiscreteMaskedPrior(num_classes=vocab)
mdlm = MDLM(
    time_distribution=UniformTD(nsteps=16),
    prior_distribution=prior,
    noise_schedule=LogLinearExpNoiseTransform(),
)
x = prior.sample((batch, seq_len))

def model_fn(x, t):
    return torch.randn(x.shape[0], x.shape[1], vocab, device=x.device)

linear_sched = LinearTimeSchedule(nsteps=16, min_t=0.0, max_t=1.0)
cos_sched = CosineTimeSchedule(nsteps=16, min_t=0.0, max_t=1.0)

x_linear = run_inference_loop(mdlm, model_fn, x, linear_sched, strategy="step")
x_cos = run_inference_loop(mdlm, model_fn, x, cos_sched, strategy="step")
```

## Notes

- Noise schedules (`LogLinearExpNoiseTransform`, `CosineNoiseTransform`, `LinearNoiseTransform`) expect time `t` in `[0, 1]`.
- `UniformTD`/`AntitheticUniformTD`/`SymmetricUniformTD` sample discrete step indices in `[0, nsteps)`.
