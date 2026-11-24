"""Autoregressive decoding, bounded response caching and inference metrics."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = False
    eos_token_id: int | None = None
    seed: int = 42

    def validate(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")


class GenerationCache:
    """Bounded least-recently-used cache for complete prompt/config results."""

    def __init__(self, maximum_entries: int = 128) -> None:
        if maximum_entries < 1:
            raise ValueError("maximum_entries must be positive")
        self.maximum_entries = maximum_entries
        self._items: OrderedDict[tuple[object, ...], tuple[int, ...]] = OrderedDict()

    def get(self, key: tuple[object, ...]) -> tuple[int, ...] | None:
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key: tuple[object, ...], value: tuple[int, ...]) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.maximum_entries:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


def _sample_next_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    generator: torch.Generator,
) -> int:
    scores = logits.float() / config.temperature
    if config.top_k:
        count = min(config.top_k, scores.numel())
        threshold = torch.topk(scores, count).values[-1]
        scores = scores.masked_fill(scores < threshold, float("-inf"))

    probabilities = torch.softmax(scores, dim=-1)
    if config.top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs >= config.top_p
        sorted_probs[remove] = 0
        filtered = torch.zeros_like(probabilities)
        filtered.scatter_(0, sorted_indices, sorted_probs)
        probabilities = filtered / filtered.sum()

    token = torch.multinomial(probabilities, 1, generator=generator)
    return int(token.item())


@torch.inference_mode()
def generate_tokens(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: GenerationConfig | None = None,
) -> torch.Tensor:
    settings = config or GenerationConfig()
    settings.validate()
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Generation expects input_ids with shape [1, sequence]")
    if input_ids.shape[1] == 0:
        raise ValueError("Generation prompt cannot be empty")

    device = next(model.parameters()).device
    generated = input_ids.to(device)
    context_length = getattr(
        getattr(model, "config", None), "context_length", generated.shape[1]
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed)

    was_training = model.training
    model.eval()
    try:
        for _ in range(settings.max_new_tokens):
            context = generated[:, -context_length:]
            logits = model(context)
            next_logits = logits[0, -1, :]
            if settings.do_sample:
                next_id = _sample_next_token(next_logits, settings, generator)
            else:
                next_id = int(torch.argmax(next_logits).item())
            next_tensor = torch.tensor([[next_id]], device=device, dtype=torch.long)
            generated = torch.cat((generated, next_tensor), dim=1)
            if settings.eos_token_id is not None and next_id == settings.eos_token_id:
                break
    finally:
        model.train(was_training)
    return generated


def benchmark_generation(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: GenerationConfig | None = None,
    *,
    repeats: int = 3,
) -> dict[str, float | int]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    settings = config or GenerationConfig()
    durations = []
    generated_count = 0
    for _ in range(repeats):
        started = time.perf_counter()
        output = generate_tokens(model, input_ids, settings)
        durations.append(time.perf_counter() - started)
        generated_count = output.shape[1] - input_ids.shape[1]
    elapsed = sum(durations) / len(durations)
    return {
        "repeats": repeats,
        "generated_tokens": generated_count,
        "mean_latency_seconds": elapsed,
        "tokens_per_second": generated_count / max(elapsed, 1e-12),
    }
