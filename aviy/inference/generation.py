"""Autoregressive decoding, bounded response caching and inference metrics."""

from __future__ import annotations

import math
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
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a non-negative integer")

        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("temperature must be a finite positive number")

        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k < 0
        ):
            raise ValueError("top_k must be a non-negative integer")

        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not math.isfinite(self.top_p)
            or not 0.0 < self.top_p <= 1.0
        ):
            raise ValueError("top_p must be finite and in (0, 1]")

        if not isinstance(self.do_sample, bool):
            raise TypeError("do_sample must be a boolean")

        if self.eos_token_id is not None and (
            isinstance(self.eos_token_id, bool)
            or not isinstance(self.eos_token_id, int)
            or self.eos_token_id < 0
        ):
            raise ValueError("eos_token_id must be a non-negative integer or None")

        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")


class GenerationCache:
    """Bounded least-recently-used cache for complete prompt/config results."""

    def __init__(self, maximum_entries: int = 128) -> None:
        if (
            isinstance(maximum_entries, bool)
            or not isinstance(maximum_entries, int)
            or maximum_entries < 1
        ):
            raise ValueError("maximum_entries must be a positive integer")
        self.maximum_entries = maximum_entries
        self._items: OrderedDict[tuple[object, ...], tuple[int, ...]] = OrderedDict()

    @staticmethod
    def _validate_key(key: tuple[object, ...]) -> None:
        if not isinstance(key, tuple):
            raise TypeError("cache key must be a tuple")
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError("cache key must contain only hashable values") from exc

    @staticmethod
    def _validate_value(value: tuple[int, ...]) -> None:
        if not isinstance(value, tuple):
            raise TypeError("cache value must be a tuple of token IDs")
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            for token in value
        ):
            raise ValueError("cache values must contain non-negative integer token IDs")

    def get(self, key: tuple[object, ...]) -> tuple[int, ...] | None:
        self._validate_key(key)
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key: tuple[object, ...], value: tuple[int, ...]) -> None:
        self._validate_key(key)
        self._validate_value(value)
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
    if torch.isnan(scores).any() or torch.isposinf(scores).any():
        raise ValueError("Sampling scores must not contain NaN or positive infinity")
    if not torch.isfinite(scores).any():
        raise ValueError("Sampling requires at least one finite candidate")

    if config.top_k:
        count = min(config.top_k, scores.numel())
        top_values, top_indices = torch.topk(scores, count)
        filtered_scores = torch.full_like(scores, float("-inf"))
        filtered_scores.scatter_(0, top_indices, top_values)
        scores = filtered_scores

    probabilities = torch.softmax(scores, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("Sampling probabilities must be finite")

    if config.top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(
            probabilities, descending=True
        )
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs >= config.top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)

        filtered = torch.zeros_like(probabilities)
        filtered.scatter_(0, sorted_indices, sorted_probs)
        total = filtered.sum()

        if not torch.isfinite(total) or total <= 0:
            raise ValueError("Top-p filtering produced invalid probabilities")

        probabilities = filtered / total

    if not torch.isfinite(probabilities).all():
        raise ValueError("Filtered sampling probabilities must be finite")

    token = torch.multinomial(probabilities, 1, generator=generator)
    return int(token.item())


@torch.inference_mode()
def generate_tokens(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: GenerationConfig | None = None,
    *,
    allowed_vocab_size: int | None = None,
) -> torch.Tensor:
    settings = config or GenerationConfig()
    settings.validate()

    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("input_ids must contain integer token IDs")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Generation expects input_ids with shape [1, sequence]")
    if input_ids.shape[1] == 0:
        raise ValueError("Generation prompt cannot be empty")

    if allowed_vocab_size is not None and (
        isinstance(allowed_vocab_size, bool)
        or not isinstance(allowed_vocab_size, int)
        or allowed_vocab_size < 1
    ):
        raise ValueError("allowed_vocab_size must be a positive integer")

    if (
        allowed_vocab_size is not None
        and settings.eos_token_id is not None
        and settings.eos_token_id >= allowed_vocab_size
    ):
        raise ValueError("eos_token_id must be inside the allowed vocabulary")

    if allowed_vocab_size is not None and (
        torch.any(input_ids < 0) or torch.any(input_ids >= allowed_vocab_size)
    ):
        raise ValueError("input_ids contain token IDs outside the allowed vocabulary")

    device = next(model.parameters()).device
    generated = input_ids.to(device=device, dtype=torch.long)

    context_length = getattr(
        getattr(model, "config", None), "context_length", generated.shape[1]
    )
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length < 1
    ):
        raise ValueError("Model context_length must be a positive integer")

    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed)

    was_training = model.training
    model.eval()
    try:
        for _ in range(settings.max_new_tokens):
            context = generated[:, -context_length:]
            logits = model(context)

            if not isinstance(logits, torch.Tensor):
                raise TypeError("Model output must be a logits tensor")
            if logits.ndim != 3 or logits.shape[:2] != context.shape:
                raise ValueError(
                    "Model logits must have shape [batch, sequence, vocabulary]"
                )
            if not logits.is_floating_point():
                raise TypeError("Model logits must have a floating-point dtype")
            if logits.shape[-1] < 1:
                raise ValueError("Model output vocabulary cannot be empty")

            next_logits = logits[0, -1, :]

            if not torch.isfinite(next_logits).all():
                raise ValueError("Model produced non-finite next-token logits")

            if allowed_vocab_size is not None:
                if allowed_vocab_size > next_logits.numel():
                    raise ValueError(
                        "allowed_vocab_size exceeds the model output vocabulary"
                    )
                invalid_positions = torch.arange(
                    next_logits.numel(), device=next_logits.device
                ) >= allowed_vocab_size
                next_logits = next_logits.masked_fill(
                    invalid_positions, float("-inf")
                )

            if settings.do_sample:
                next_id = _sample_next_token(next_logits, settings, generator)
            else:
                next_id = int(torch.argmax(next_logits).item())

            next_tensor = torch.tensor(
                [[next_id]], device=device, dtype=torch.long
            )
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
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")

    settings = config or GenerationConfig()
    settings.validate()

    device = next(model.parameters()).device
    durations: list[float] = []
    generated_count = 0

    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        started = time.perf_counter()
        output = generate_tokens(model, input_ids, settings)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        duration = time.perf_counter() - started
        if not math.isfinite(duration) or duration < 0:
            raise RuntimeError("Benchmark produced an invalid duration")

        durations.append(duration)
        generated_count = output.shape[1] - input_ids.shape[1]

    elapsed = sum(durations) / len(durations)
    return {
        "repeats": repeats,
        "generated_tokens": generated_count,
        "mean_latency_seconds": elapsed,
        "tokens_per_second": generated_count / max(elapsed, 1e-12),
    }
