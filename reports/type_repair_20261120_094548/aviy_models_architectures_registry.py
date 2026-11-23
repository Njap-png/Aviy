"""Architecture factory backed by Aviy's validated registry."""
from __future__ import annotations

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

def build_model(name: str, config: ArchitectureConfig | None = None, **kwargs: object) -> LanguageModel:
    if config is not None and kwargs:
        raise ValueError("Pass either config or configuration keyword arguments")
    resolved = config or ArchitectureConfig(**kwargs)
    return ARCHITECTURES.create(name, resolved)
