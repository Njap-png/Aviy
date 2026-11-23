"""Reproducible tokenizer evaluation and report generation."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

from aviy.tokenizer.bpe import ByteBPETokenizer, TokenizerMetrics


class TokenizerBenchmark(TypedDict):
    format: str
    tokenizer: str
    vocabulary_size: int
    normalization: str
    coverage_definition: str
    metrics: TokenizerMetrics
    limitations: list[str]


def benchmark_tokenizer(
    tokenizer: ByteBPETokenizer,
    texts: Iterable[str],
) -> TokenizerBenchmark:
    """Measure actual byte/character coverage and round-trip behavior."""
    samples = list(texts)
    metrics = tokenizer.measure(samples)

    return {
        "format": "aviy-tokenizer-benchmark-v1",
        "tokenizer": "byte-level-bpe",
        "vocabulary_size": tokenizer.vocabulary_size,
        "normalization": "Unicode NFC and whitespace normalization",
        "coverage_definition": (
            "Measured by comparing normalized input with decoded encoded output."
        ),
        "metrics": metrics,
        "limitations": [
            "Byte-level encoding represents valid Unicode input without an unknown token.",
            "Round-trip measurements detect implementation or serialization errors.",
            "Fertility depends on the evaluated corpus.",
            "A synthetic benchmark is not evidence of real-corpus quality.",
        ],
    }


def write_tokenizer_benchmark(
    tokenizer: ByteBPETokenizer,
    texts: Iterable[str],
    path: str | Path,
) -> TokenizerBenchmark:
    report = benchmark_tokenizer(tokenizer, texts)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
