"""Planning, step verification and reasoning reliability measurements."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    description: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReasoningResult:
    answer: str
    steps: tuple[PlanStep, ...]
    verified: tuple[bool, ...]
    confidence: float
    correctness: bool | None
    failed_step_ids: tuple[str, ...]
    brier_score: float | None


class Planner:
    def plan(self, steps: Sequence[PlanStep]) -> tuple[PlanStep, ...]:
        if not steps:
            raise ValueError("A plan must contain at least one step")
        by_id = {step.step_id: step for step in steps}
        if len(by_id) != len(steps) or any(not step.step_id.strip() for step in steps):
            raise ValueError("Plan step IDs must be non-empty and unique")
        for step in steps:
            if not step.description.strip():
                raise ValueError("Plan step descriptions cannot be empty")
            if any(dependency not in by_id for dependency in step.dependencies):
                raise ValueError("Plan contains an unknown dependency")

        ordered: list[PlanStep] = []
        remaining = dict(by_id)
        completed: set[str] = set()
        while remaining:
            ready = [
                step for step in remaining.values()
                if set(step.dependencies).issubset(completed)
            ]
            if not ready:
                raise ValueError("Plan contains a dependency cycle")
            ready.sort(key=lambda step: step.step_id)
            for step in ready:
                ordered.append(step)
                completed.add(step.step_id)
                del remaining[step.step_id]
        return tuple(ordered)


def evaluate_reasoning(
    steps: Sequence[PlanStep],
    answer: str,
    verifier: Callable[[PlanStep], bool],
    *,
    confidence: float,
    expected_correctness: bool | None = None,
) -> ReasoningResult:
    if not 0.0 <= confidence <= 1.0 or not math.isfinite(confidence):
        raise ValueError("confidence must be finite and in [0, 1]")
    planned = Planner().plan(steps)
    verified = tuple(bool(verifier(step)) for step in planned)
    failed = tuple(
        step.step_id for step, passed in zip(planned, verified) if not passed
    )
    correctness = (
        all(verified) if expected_correctness is None else expected_correctness
    )
    brier = (
        (confidence - float(correctness)) ** 2
        if expected_correctness is not None else None
    )
    return ReasoningResult(
        answer=answer,
        steps=planned,
        verified=verified,
        confidence=confidence,
        correctness=correctness,
        failed_step_ids=failed,
        brier_score=brier,
    )


def detect_reasoning_failure(result: ReasoningResult) -> dict[str, object]:
    return {
        "passed": not result.failed_step_ids,
        "failed_step_ids": list(result.failed_step_ids),
        "verification_rate": (
            sum(result.verified) / len(result.verified) if result.verified else 0.0
        ),
        "confidence": result.confidence,
        "brier_score": result.brier_score,
    }
