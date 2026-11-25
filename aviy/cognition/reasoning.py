"""Planning, step verification and reasoning reliability measurements."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PlanStep:
    """A single reasoning step and its prerequisite step identifiers."""

    step_id: str
    description: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReasoningResult:
    """Structured result of a verified reasoning execution."""

    answer: str
    steps: tuple[PlanStep, ...]
    verified: tuple[bool, ...]
    confidence: float
    correctness: bool | None
    failed_step_ids: tuple[str, ...]
    brier_score: float | None


class ReasoningPlanner(Protocol):
    """Interface implemented by reasoning planners."""

    def plan(self, steps: Sequence[PlanStep]) -> tuple[PlanStep, ...]:
        """Return steps in dependency-safe execution order."""
        ...


class StepVerifier(Protocol):
    """Interface for deterministic or externally supplied step verification."""

    def __call__(self, step: PlanStep) -> bool:
        """Return whether a reasoning step passed verification."""
        ...


class Planner:
    """Validate plans and produce deterministic dependency ordering."""

    def plan(self, steps: Sequence[PlanStep]) -> tuple[PlanStep, ...]:
        if (
            not isinstance(steps, Sequence)
            or isinstance(steps, (str, bytes))
            or not steps
        ):
            raise ValueError("A plan must contain at least one step")

        by_id: dict[str, PlanStep] = {}

        for step in steps:
            if not isinstance(step, PlanStep):
                raise TypeError("Every plan entry must be a PlanStep")

            if not isinstance(step.step_id, str) or not step.step_id.strip():
                raise ValueError("Plan step IDs must be non-empty strings")

            if step.step_id in by_id:
                raise ValueError("Plan step IDs must be unique")

            if not isinstance(step.description, str) or not step.description.strip():
                raise ValueError("Plan step descriptions cannot be empty")

            if not isinstance(step.dependencies, tuple):
                raise TypeError("Plan dependencies must be a tuple")

            if any(
                not isinstance(dependency, str) or not dependency.strip()
                for dependency in step.dependencies
            ):
                raise ValueError("Dependencies must be non-empty strings")

            if len(set(step.dependencies)) != len(step.dependencies):
                raise ValueError("Duplicate dependencies are not allowed")

            if step.step_id in step.dependencies:
                raise ValueError("A plan step cannot depend on itself")

            by_id[step.step_id] = step

        for step in steps:
            if any(dependency not in by_id for dependency in step.dependencies):
                raise ValueError("Plan contains an unknown dependency")

        ordered: list[PlanStep] = []
        remaining = dict(by_id)
        completed: set[str] = set()

        while remaining:
            ready = [
                step
                for step in remaining.values()
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
    verifier: StepVerifier,
    *,
    confidence: float,
    expected_correctness: bool | None = None,
) -> ReasoningResult:
    """Plan and verify reasoning steps, then calculate calibration evidence."""

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")

    if not callable(verifier):
        raise TypeError("verifier must be callable")

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
    ):
        raise ValueError("confidence must be finite and in [0, 1]")

    if expected_correctness is not None and type(expected_correctness) is not bool:
        raise TypeError("expected_correctness must be bool or None")

    planned = Planner().plan(steps)
    verification_results: list[bool] = []

    for step in planned:
        outcome = verifier(step)

        if type(outcome) is not bool:
            raise TypeError("Verifier must return an actual bool")

        verification_results.append(outcome)

    verified = tuple(verification_results)
    failed = tuple(
        step.step_id
        for step, passed in zip(planned, verified, strict=True)
        if not passed
    )

    correctness = (
        all(verified)
        if expected_correctness is None
        else expected_correctness
    )

    brier = (
        (float(confidence) - float(correctness)) ** 2
        if expected_correctness is not None
        else None
    )

    return ReasoningResult(
        answer=answer,
        steps=planned,
        verified=verified,
        confidence=float(confidence),
        correctness=correctness,
        failed_step_ids=failed,
        brier_score=brier,
    )


def detect_reasoning_failure(result: ReasoningResult) -> dict[str, object]:
    """Produce structured verification and uncertainty diagnostics."""

    if not isinstance(result, ReasoningResult):
        raise TypeError("result must be a ReasoningResult")

    if len(result.steps) != len(result.verified):
        raise ValueError("Reasoning result steps and verification values must match")

    if any(type(value) is not bool for value in result.verified):
        raise ValueError("Verification values must be booleans")

    expected_failures = tuple(
        step.step_id
        for step, passed in zip(result.steps, result.verified, strict=True)
        if not passed
    )

    if result.failed_step_ids != expected_failures:
        raise ValueError("Reasoning result contains inconsistent failure identifiers")

    if (
        isinstance(result.confidence, bool)
        or not isinstance(result.confidence, (int, float))
        or not math.isfinite(result.confidence)
        or not 0.0 <= result.confidence <= 1.0
    ):
        raise ValueError("Reasoning result confidence is invalid")

    if result.brier_score is not None and (
        isinstance(result.brier_score, bool)
        or not isinstance(result.brier_score, (int, float))
        or not math.isfinite(result.brier_score)
        or not 0.0 <= result.brier_score <= 1.0
    ):
        raise ValueError("Reasoning result Brier score is invalid")

    return {
        "passed": not result.failed_step_ids,
        "failed_step_ids": list(result.failed_step_ids),
        "verification_rate": (
            sum(result.verified) / len(result.verified)
            if result.verified
            else 0.0
        ),
        "confidence": result.confidence,
        "correctness": result.correctness,
        "brier_score": result.brier_score,
    }
