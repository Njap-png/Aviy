"""Reproducible tokenizer evaluation and report generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

from aviy.tokenizer.bpe import ByteBPETokenizer, TokenizerMetrics


class TokenizerBenchmark(TypedDict):
    format: str
    tokenizer: str
    vocabulary_size: int
    tokenizer_sha256: str
    corpus_sha256: str
    sample_count: int
    normalization: str
    coverage_definition: str
    metrics: TokenizerMetrics
    limitations: list[str]


def benchmark_tokenizer(
    tokenizer: ByteBPETokenizer,
    texts: Iterable[str],
) -> TokenizerBenchmark:
    """Measure tokenizer behavior and identify the exact inputs evaluated."""
    samples = [tokenizer.normalize(text) for text in texts]
    metrics = tokenizer.measure(samples)

    tokenizer_payload = json.dumps(
        {
            "format": "aviy-byte-bpe-v1",
            "merges": tokenizer.merges,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    corpus_payload = json.dumps(
        samples,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return {
        "format": "aviy-tokenizer-benchmark-v2",
        "tokenizer": "byte-level-bpe",
        "vocabulary_size": tokenizer.vocabulary_size,
        "tokenizer_sha256": hashlib.sha256(tokenizer_payload).hexdigest(),
        "corpus_sha256": hashlib.sha256(corpus_payload).hexdigest(),
        "sample_count": len(samples),
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
            "Hashes identify the measured inputs but do not establish dataset licensing or authenticity.",
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
