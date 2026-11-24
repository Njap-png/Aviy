"""Architecture factory backed by Aviy's validated registry."""

from __future__ import annotations

from typing import Any, cast

from aviy.core.registry import Registry
from aviy.models.architectures.base import ArchitectureConfig, LanguageModel
from aviy.models.architectures.families import (
    GNNLM,
    RWKVLM,
    SSMLM,
    ConvLM,
    HybridLM,
    MoELM,
    RecurrentLM,
    TransformerLM,
)


def create_architecture_registry() -> Registry:
    registry = Registry("model_architectures")
    for name, factory in {
        "transformer": TransformerLM,
        "rnn": lambda config: RecurrentLM(config, "rnn"),
        "gru": lambda config: RecurrentLM(config, "gru"),
        "lstm": lambda config: RecurrentLM(config, "lstm"),
        "ssm": SSMLM,
        "rwkv": RWKVLM,
        "cnn": ConvLM,
        "moe": MoELM,
        "gnn": GNNLM,
        "hybrid": HybridLM,
    }.items():
        registry.register(name, factory)
    return registry


ARCHITECTURES = create_architecture_registry()


def build_model(
    name: str,
    config: ArchitectureConfig | None = None,
    **kwargs: object,
) -> LanguageModel:
    """Build a registered architecture from a config or validated config fields."""
    if config is not None and kwargs:
        raise ValueError("Pass either config or configuration keyword arguments")

    if config is None:
        allowed = set(ArchitectureConfig.__dataclass_fields__)
        unknown = set(kwargs) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown architecture configuration fields: {names}")
        for key, value in kwargs.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"Architecture configuration field {key!r} must be numeric"
                )
        resolved = ArchitectureConfig(**cast(Any, kwargs))
    else:
        resolved = config

    return ARCHITECTURES.create(name, resolved)
