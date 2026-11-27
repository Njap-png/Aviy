"""Synthetic engineering evaluation for reasoning verification and calibration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from aviy.cognition.reasoning import (
    PlanStep,
    detect_reasoning_failure,
    evaluate_reasoning,
)


def build_synthetic_reasoning_report() -> dict[str, object]:
    """Build reproducible synthetic evidence without claiming model capability."""

    scenarios = (
        {
            "case_id": "dependency-order",
            "steps": (
                PlanStep("validate", "Validate the result", ("calculate",)),
                PlanStep("calculate", "Calculate the result"),
            ),
            "answer": "Synthetic calculation passed",
            "verifier": lambda _: True,
            "confidence": 0.95,
            "expected_correctness": True,
        },
        {
            "case_id": "failed-verification",
            "steps": (
                PlanStep("calculate", "Calculate the result"),
                PlanStep("validate", "Validate the result"),
            ),
            "answer": "Synthetic calculation failed validation",
            "verifier": lambda step: step.step_id == "calculate",
            "confidence": 0.8,
            "expected_correctness": False,
        },
        {
            "case_id": "uncertain-answer",
            "steps": (PlanStep("check", "Check the answer"),),
            "answer": "Synthetic uncertain answer",
            "verifier": lambda _: True,
            "confidence": 0.5,
            "expected_correctness": True,
        },
    )

    cases: list[dict[str, object]] = []

    for scenario in scenarios:
        result = evaluate_reasoning(
            scenario["steps"],
            scenario["answer"],
            scenario["verifier"],
            confidence=scenario["confidence"],
            expected_correctness=scenario["expected_correctness"],
        )

        diagnostics = detect_reasoning_failure(result)

        cases.append(
            {
                "case_id": scenario["case_id"],
                "confidence": result.confidence,
                "expected_correctness": result.correctness,
                "verified_steps": list(result.verified),
                "failed_step_ids": list(result.failed_step_ids),
                "verification": diagnostics,
                "brier_score": result.brier_score,
            }
        )

    scores = [
        float(score)
        for case in cases
        if isinstance((score := case["brier_score"]), (int, float))
    ]

    return {
        "format": "aviy-reasoning-engineering-evaluation-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evidence_class": "synthetic_engineering_fixture",
        "model_capability_claim": False,
        "cases": cases,
        "metrics": {
            "case_count": len(cases),
            "mean_brier_score": mean(scores),
            "all_cases_have_verification": all(
                isinstance(case["verification"], dict)
                for case in cases
            ),
            "failed_case_count": sum(
                not bool(verification["passed"])
                for case in cases
                if isinstance(
                    (verification := case["verification"]), dict
                )
            ),
            "correctness_labels_present": all(
                isinstance(case["expected_correctness"], bool)
                for case in cases
            ),
            "confidence_values_valid": all(
                isinstance(
                    (confidence := case["confidence"]), (int, float)
                )
                and 0.0 <= float(confidence) <= 1.0
                for case in cases
            ),
        },
        "limitations": [
            "Synthetic verifier outcomes; no language model reasoning was evaluated.",
            "This report validates the reasoning interface and scoring behavior only.",
            "It is not evidence of real-world reasoning accuracy or calibration.",
        ],
    }


def write_synthetic_reasoning_report(path: str | Path) -> dict[str, object]:
    """Generate and persist a synthetic reasoning evaluation report."""

    report = build_synthetic_reasoning_report()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report
