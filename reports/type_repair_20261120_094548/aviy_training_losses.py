"""Language-model losses and regularization helpers."""
from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def causal_language_model_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Cross-entropy for next-token prediction."""
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("Expected logits [B,T,V] and targets [B,T]")
    if logits.shape[:2] != targets.shape:
        raise ValueError("Logit and target batch/sequence shapes differ")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1).long(),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


def l2_regularization(parameters: object, coefficient: float) -> Tensor:
    """Compute explicit L2 penalty over trainable tensors."""
    if coefficient < 0:
        raise ValueError("coefficient cannot be negative")
    tensors = [p for p in parameters if p.requires_grad]
    if not tensors:
        raise ValueError("No trainable parameters supplied")
    return coefficient * sum(p.float().pow(2).sum() for p in tensors)
