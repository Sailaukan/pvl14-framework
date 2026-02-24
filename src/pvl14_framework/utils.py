import torch
from torch import Tensor


def pad_like(source: Tensor, target: Tensor) -> Tensor:
    if source.ndim == target.ndim:
        return source
    if source.ndim > target.ndim:
        raise ValueError(f"Cannot pad {source.shape} to {target.shape}")
    return source.view(list(source.shape) + [1] * (target.ndim - source.ndim))
