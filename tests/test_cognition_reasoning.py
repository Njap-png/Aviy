from __future__ import annotations

import pytest

from aviy.cognition.evaluation import (
    build_synthetic_reasoning_report,
    write_synthetic_reasoning_report,
)
from aviy.cognition.reasoning import (
    Planner,
    PlanStep,
    ReasoningResult,
    detect_reasoning_failure,
    evaluate_reasoning,
)


def test_planner_orders_dependencies_deterministically():
    steps = (
        PlanStep("third", "Third", ("second",)),
        PlanStep("second", "Second", ("first",)),
        PlanStep("first", "First"),
    )

    assert [
        step.step_id for step in Planner().plan(steps)
    ] == ["first", "second", "third"]


@pytest.mark.parametrize(
    "steps",
    [
        (),
        (PlanStep("", "Description"),),
        (PlanStep("a", ""),),
        (PlanStep("a", "A"), PlanStep("a", "Duplicate")),
        (PlanStep("a", "A", ("missing",)),),
        (PlanStep("a", "A", ("a",)),),
        (PlanStep("a", "A", ("b",)), PlanStep("b", "B", ("a",))),
        (PlanStep("a", "A", ("b", "b")), PlanStep("b", "B")),
    ],
)
def test_planner_rejects_invalid_plans(steps):
    with pytest.raises(ValueError):
        Planner().plan(steps)


def test_planner_rejects_invalid_step_types():
    with pytest.raises(TypeError, match="PlanStep"):
        Planner().plan((object(),))


def test_planner_rejects_invalid_dependency_container():
    with pytest.raises(TypeError, match="tuple"):
        Planner().plan((PlanStep("a", "A", ["b"]), PlanStep("b", "B")))


def test_reasoning_verifies_steps_and_calculates_brier_score():
    result = evaluate_reasoning(
        (
            PlanStep("second", "Second", ("first",)),
            PlanStep("first", "First"),
        ),
        "Verified answer",
        lambda step: step.step_id == "first",
        confidence=0.8,
        expected_correctness=False,
    )

    assert result.verified == (True, False)
    assert result.failed_step_ids == ("second",)
    assert result.correctness is False
    assert result.brier_score == pytest.approx(0.64)

    diagnostics = detect_reasoning_failure(result)
    assert diagnostics["passed"] is False
    assert diagnostics["verification_rate"] == pytest.approx(0.5)


def test_reasoning_without_ground_truth_does_not_claim_calibration():
    result = evaluate_reasoning(
        (PlanStep("a", "Check"),),
        "Answer",
        lambda _: True,
        confidence=0.7,
    )

    assert result.correctness is True
    assert result.brier_score is None


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_reasoning_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        evaluate_reasoning(
            (PlanStep("a", "Check"),),
            "Answer",
            lambda _: True,
            confidence=confidence,
        )


def test_reasoning_rejects_invalid_answer_and_verifier():
    with pytest.raises(ValueError, match="answer"):
        evaluate_reasoning(
            (PlanStep("a", "Check"),), "", lambda _: True, confidence=0.5
        )

    with pytest.raises(TypeError, match="callable"):
        evaluate_reasoning(
            (PlanStep("a", "Check"),), "Answer", None, confidence=0.5
        )


@pytest.mark.parametrize("outcome", [1, 0, "yes", None, object()])
def test_reasoning_verifier_must_return_actual_boolean(outcome):
    with pytest.raises(TypeError, match="actual bool"):
        evaluate_reasoning(
            (PlanStep("a", "Check"),),
            "Answer",
            lambda _: outcome,
            confidence=0.5,
        )


def test_reasoning_rejects_non_boolean_ground_truth():
    with pytest.raises(TypeError, match="expected_correctness"):
        evaluate_reasoning(
            (PlanStep("a", "Check"),),
            "Answer",
            lambda _: True,
            confidence=0.5,
            expected_correctness=1,
        )


def test_failure_detection_rejects_inconsistent_results():
    invalid = ReasoningResult(
        answer="Answer",
        steps=(PlanStep("a", "Check"),),
        verified=(True,),
        confidence=0.5,
        correctness=True,
        failed_step_ids=("a",),
        brier_score=None,
    )

    with pytest.raises(ValueError, match="inconsistent"):
        detect_reasoning_failure(invalid)


def test_synthetic_report_measures_correctness_and_uncertainty():
    report = build_synthetic_reasoning_report()

    assert report["evidence_class"] == "synthetic_engineering_fixture"
    assert report["model_capability_claim"] is False
    assert report["metrics"]["case_count"] == 3
    assert report["metrics"]["all_cases_have_verification"] is True
    assert report["metrics"]["correctness_labels_present"] is True
    assert report["metrics"]["confidence_values_valid"] is True
    assert report["metrics"]["failed_case_count"] == 1
    assert report["metrics"]["mean_brier_score"] == pytest.approx(0.2975)


def test_synthetic_report_can_be_written_and_reloaded(tmp_path):
    import json

    destination = tmp_path / "reasoning" / "evaluation.json"
    report = write_synthetic_reasoning_report(destination)
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded["metrics"] == report["metrics"]
    assert loaded["model_capability_claim"] is False
