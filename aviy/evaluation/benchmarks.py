"""Capability-specific evaluation, regression and contamination checks."""
from __future__ import annotations

import hashlib
import json
import math
import time
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

EVALUATION_DIMENSIONS = (
    "knowledge_accuracy", "reasoning", "instruction_following",
    "language_generation", "programming", "mathematics",
    "scientific_understanding", "multilingual", "factuality",
    "uncertainty_calibration", "safety", "efficiency",
)


def normalize_for_comparison(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    dimension: str
    prompt: str
    expected: str


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    cases: tuple[BenchmarkCase, ...]

    def validate(self, require_all_dimensions: bool = True) -> None:
        if not self.name.strip() or not self.cases:
            raise ValueError("Benchmark suite requires a name and cases")
        ids: set[str] = set()
        dimensions: set[str] = set()
        for case in self.cases:
            if not case.case_id.strip() or case.case_id in ids:
                raise ValueError("Benchmark case IDs must be non-empty and unique")
            if case.dimension not in EVALUATION_DIMENSIONS:
                raise ValueError(f"Unsupported evaluation dimension: {case.dimension}")
            if not case.prompt.strip():
                raise ValueError("Benchmark prompts cannot be empty")
            ids.add(case.case_id)
            dimensions.add(case.dimension)
        if require_all_dimensions:
            missing = set(EVALUATION_DIMENSIONS) - dimensions
            if missing:
                raise ValueError(
                    "Benchmark suite is missing dimensions: "
                    + ", ".join(sorted(missing))
                )


def check_contamination(
    suite: BenchmarkSuite, training_texts: Iterable[str]
) -> dict[str, object]:
    """Detect exact normalized prompt or expected-answer overlap."""
    training = {normalize_for_comparison(item) for item in training_texts}
    overlaps = [
        case.case_id for case in suite.cases
        if normalize_for_comparison(case.prompt) in training
        or normalize_for_comparison(case.expected) in training
    ]
    return {
        "passed": not overlaps,
        "overlapping_case_ids": overlaps,
        "training_fingerprints": len(training),
        "method": "exact_normalized_match",
    }


def run_benchmark(
    suite: BenchmarkSuite,
    predictor: Callable[[str], str],
    *,
    seed: int = 42,
) -> dict[str, object]:
    """Run exact-match evaluation with all 12 dimension reports."""
    suite.validate()
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    groups: dict[str, list[bool]] = defaultdict(list)
    outputs: list[dict[str, object]] = []
    started = time.perf_counter()

    for case in suite.cases:
        prediction = predictor(case.prompt)
        if not isinstance(prediction, str):
            raise TypeError("Predictor must return strings")
        correct = (
            normalize_for_comparison(prediction)
            == normalize_for_comparison(case.expected)
        )
        groups[case.dimension].append(correct)
        outputs.append({
            "case_id": case.case_id,
            "dimension": case.dimension,
            "correct": correct,
            "prediction": prediction,
        })

    elapsed = max(time.perf_counter() - started, 1e-12)
    dimension_metrics = {
        dimension: {
            "cases": len(groups[dimension]),
            "accuracy": (
                sum(groups[dimension]) / len(groups[dimension])
                if groups[dimension] else None
            ),
            "reliability": (
                sum(groups[dimension]) / len(groups[dimension])
                if groups[dimension] else None
            ),
        }
        for dimension in EVALUATION_DIMENSIONS
    }
    correct_count = sum(bool(item["correct"]) for item in outputs)
    canonical = json.dumps(
        {"suite": suite.name, "seed": seed, "outputs": outputs},
        sort_keys=True,
        ensure_ascii=False,
    )
    return {
        "format": "aviy-benchmark-v1",
        "suite": suite.name,
        "seed": seed,
        "cases": len(outputs),
        "accuracy": correct_count / len(outputs),
        "reliability": correct_count / len(outputs),
        "elapsed_seconds": elapsed,
        "cases_per_second": len(outputs) / elapsed,
        "dimensions": dimension_metrics,
        "outputs": outputs,
        "reproducibility_fingerprint": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
        "metric_note": "Reliability is exact-match success rate in this baseline.",
    }


def compare_regression(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    minimum_accuracy: float = 0.0,
    maximum_drop: float = 0.02,
) -> dict[str, object]:
    """Compare benchmark accuracy and reject regressions beyond a threshold."""
    for value in (minimum_accuracy, maximum_drop):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("Regression thresholds must be finite and in [0, 1]")
    old_value = baseline.get("accuracy")
    new_value = candidate.get("accuracy")
    if (
        isinstance(old_value, bool) or not isinstance(old_value, (int, float))
        or isinstance(new_value, bool) or not isinstance(new_value, (int, float))
        or not math.isfinite(old_value) or not math.isfinite(new_value)
        or not 0.0 <= old_value <= 1.0 or not 0.0 <= new_value <= 1.0
    ):
        raise ValueError("Benchmark accuracy values must be finite numbers in [0, 1]")
    old = float(old_value)
    new = float(new_value)
    drop = old - new
    return {
        "passed": new >= minimum_accuracy and drop <= maximum_drop,
        "baseline_accuracy": old,
        "candidate_accuracy": new,
        "accuracy_drop": drop,
    }


def write_benchmark_report(
    report: dict[str, object], path: str | Path
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
