![PVL14 Framework Cover](pvl14_cover.png)

# PVL14

Lightweight PyTorch code for masked discrete diffusion with MDDM-style decoding.

## Current codebase

Core classes exported by `pvl14/__init__.py`:

- `MDDM`
- `UniformTD`
- `AntitheticUniformTD`
- `SymmetricUniformTD`
- `DiscreteMaskedPrior`
- `LogLinearExpNoiseTransform`
- `LinearTimeSchedule`
- `CosineTimeSchedule`
- `ExponentialTimeSchedule`
- `run_inference_loop`

Additional schedules available from `pvl14/noise.py`:

- `CosineNoiseTransform`
- `LinearNoiseTransform`

## Installation

```bash
pip install -e .
```

## Quick start (matches current APIs)

```python
import torch
from pvl14 import (
    MDDM,
    UniformTD,
    DiscreteMaskedPrior,
    LogLinearExpNoiseTransform,
)

batch, seq_len, vocab = 2, 16, 100
mask_idx = vocab - 1

prior = DiscreteMaskedPrior(num_classes=vocab, mask_dim=mask_idx)
mddm = MDDM(
    time_distribution=UniformTD(nsteps=8),
    prior_distribution=prior,
    noise_schedule=LogLinearExpNoiseTransform(),
)

# Start from a fully masked sample
xt = prior.sample(shape=(batch, seq_len))
logits = torch.randn(batch, seq_len, vocab)

xt_next = mddm.step_confidence(
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
from pvl14 import MDDM, UniformTD, DiscreteMaskedPrior, LogLinearExpNoiseTransform

batch, seq_len, vocab = 4, 12, 64
prior = DiscreteMaskedPrior(num_classes=vocab)
mddm = MDDM(
    time_distribution=UniformTD(nsteps=16),
    prior_distribution=prior,
    noise_schedule=LogLinearExpNoiseTransform(),
)

x0 = torch.randint(0, vocab - 1, (batch, seq_len))
t = torch.rand(batch)  # continuous time in [0, 1]
xt = mddm.forward_process(x0, t)
logits = torch.randn(batch, seq_len, vocab)

loss_per_sample = mddm.loss(logits=logits, target=x0, xt=xt, time=t)
```

## Decoding options

- `decode_strategy="confidence"` (default): unmask top-confidence positions each step.
- `decode_strategy="self_path_planning"`: uses re-masking / regeneration behavior through `step_confidence(...)`.
- `decode_strategy="threshold_regen"`: re-masks low-confidence unmasked tokens and regenerates them in later steps.
  - Configure with `confidence_threshold`, `min_conf_gain`, `max_remask_frac`, and `allow_remask_unmasked`.

## Inference schedules + helper loop

```python
import torch
from pvl14 import (
    MDDM,
    UniformTD,
    DiscreteMaskedPrior,
    LogLinearExpNoiseTransform,
    LinearTimeSchedule,
    CosineTimeSchedule,
    ExponentialTimeSchedule,
    run_inference_loop,
)

batch, seq_len, vocab = 2, 16, 100
prior = DiscreteMaskedPrior(num_classes=vocab)
mddm = MDDM(
    time_distribution=UniformTD(nsteps=16),
    prior_distribution=prior,
    noise_schedule=LogLinearExpNoiseTransform(),
)
x = prior.sample((batch, seq_len))

def model_fn(x, t):
    return torch.randn(x.shape[0], x.shape[1], vocab, device=x.device)

linear_sched = LinearTimeSchedule(nsteps=16, min_t=0.0, max_t=1.0)
cos_sched = CosineTimeSchedule(nsteps=16, min_t=0.0, max_t=1.0)
exp_sched = ExponentialTimeSchedule(nsteps=16, min_t=0.0, max_t=1.0, k=5.0)

x_linear = run_inference_loop(mddm, model_fn, x, linear_sched, strategy="step")
x_cos = run_inference_loop(mddm, model_fn, x, cos_sched, strategy="step")
x_exp = run_inference_loop(mddm, model_fn, x, exp_sched, strategy="step")
```

## Notes

- Noise schedules (`LogLinearExpNoiseTransform`, `CosineNoiseTransform`, `LinearNoiseTransform`) expect time `t` in `[0, 1]`.
- `UniformTD`/`AntitheticUniformTD`/`SymmetricUniformTD` sample discrete step indices in `[0, nsteps)`.
