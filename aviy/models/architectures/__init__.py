"""Replaceable Aviy neural architecture library."""

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
from aviy.models.architectures.registry import ARCHITECTURES, build_model

__all__ = [
    "ARCHITECTURES",
    "GNNLM",
    "RWKVLM",
    "SSMLM",
    "ArchitectureConfig",
    "ConvLM",
    "HybridLM",
    "LanguageModel",
    "MoELM",
    "RecurrentLM",
    "TransformerLM",
    "build_model",
]
