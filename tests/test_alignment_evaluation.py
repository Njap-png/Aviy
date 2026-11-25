from __future__ import annotations

from dataclasses import replace

import pytest

from aviy.inference.generation import GenerationConfig
from aviy.inference.runtime import InferenceEngine
from aviy.learning.alignment import evaluate_instruction_candidate
from aviy.learning.instruction import (
    InstructionRecord,
    PreparedInstructionDataset,
    prepare_instruction_dataset,
)
from aviy.models.base_model import AviyBaseLM, BaseModelConfig
from aviy.tokenizer.bpe import ByteBPETokenizer


def make_engine() -> InferenceEngine:
    model = AviyBaseLM(
        BaseModelConfig(
            vocab_size=256,
            context_length=256,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
        )
    )
    return InferenceEngine(model, ByteBPETokenizer())


def make_dataset() -> PreparedInstructionDataset:
    records = [
        InstructionRecord(
            prompt=f"Explain concept number {index} in simple words.",
            response=f"Concept {index} has a unique explanation.",
            source_id="alignment-test",
            license="CC-BY-4.0",
            approved_for_training=True,
            source_uri="https://example.org/alignment-test",
            record_id=f"record-{index}",
        )
        for index in range(120)
    ]
    dataset = prepare_instruction_dataset(
        records,
        allowed_licenses=["CC-BY-4.0"],
        seed=42,
    )
    assert dataset.audit.passed
    assert dataset.split("train")
    assert dataset.split("test")
    return dataset


def passing_evaluator(prompt: str, generated: str) -> dict[str, float]:
    assert isinstance(prompt, str)
    assert isinstance(generated, str)
    return {"instruction_following": 0.9, "safety": 0.99}


def test_evaluation_uses_actual_inference_and_records_outputs():
    engine = make_engine()
    dataset = make_dataset()

    report = evaluate_instruction_candidate(
        engine,
        dataset,
        passing_evaluator,
        generation_config=GenerationConfig(max_new_tokens=2),
    )

    assert report["evaluation"] == "inference_backed_held_out_alignment"
    assert report["cases"] == len(dataset.split("test"))
    assert report["scope"]["uses_actual_inference"] is True
    assert report["scope"]["uses_test_split_only"] is True
    assert report["candidate"]["preference"] is None
    assert report["candidate"]["preference_measured"] is False
    assert report["regression"]["measured"] is False
    assert len(report["results"]) == report["cases"]
    assert all(isinstance(item["generated_response"], str) for item in report["results"])


def test_evaluation_can_measure_preference_and_baseline_regression():
    engine = make_engine()
    baseline = make_engine()

    report = evaluate_instruction_candidate(
        engine,
        make_dataset(),
        passing_evaluator,
        preference_evaluator=lambda prompt, answer: 0.75,
        baseline_engine=baseline,
        generation_config=GenerationConfig(max_new_tokens=1),
    )

    assert report["candidate"]["preference"] == 0.75
    assert report["candidate"]["preference_measured"] is True
    assert report["regression"]["measured"] is True
    assert report["regression"]["passed"] is True
    assert report["passed"] is True


def test_evaluation_rejects_training_overlap_in_test_split():
    dataset = make_dataset()
    training_record = dataset.split("train")[0]
    test_record = dataset.split("test")[0]
    altered_test = replace(test_record, prompt=training_record.prompt)
    records = tuple(
        altered_test if item.record_id == test_record.record_id else item
        for item in dataset.records
    )
    contaminated = PreparedInstructionDataset(records, dataset.audit)

    with pytest.raises(ValueError, match="overlaps training data"):
        evaluate_instruction_candidate(
            make_engine(),
            contaminated,
            passing_evaluator,
            generation_config=GenerationConfig(max_new_tokens=1),
        )


def test_evaluation_rejects_failed_dataset_audit():
    dataset = make_dataset()
    failed_audit = replace(dataset.audit, privacy_scan_passed=False)
    failed_dataset = PreparedInstructionDataset(dataset.records, failed_audit)

    with pytest.raises(ValueError, match="audit did not pass"):
        evaluate_instruction_candidate(make_engine(), failed_dataset, passing_evaluator)


def test_evaluation_rejects_invalid_evaluator_scores():
    with pytest.raises(ValueError, match="invalid safety"):
        evaluate_instruction_candidate(
            make_engine(),
            make_dataset(),
            lambda prompt, answer: {
                "instruction_following": 0.9,
                "safety": float("nan"),
            },
            generation_config=GenerationConfig(max_new_tokens=1),
        )


def test_evaluation_rejects_missing_score_fields():
    with pytest.raises(ValueError, match="invalid safety"):
        evaluate_instruction_candidate(
            make_engine(),
            make_dataset(),
            lambda prompt, answer: {"instruction_following": 0.9},
            generation_config=GenerationConfig(max_new_tokens=1),
        )


def test_evaluation_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="minimum_safety_score"):
        evaluate_instruction_candidate(
            make_engine(),
            make_dataset(),
            passing_evaluator,
            minimum_safety_score=1.5,
        )


def test_failed_scores_fail_candidate_evaluation():
    report = evaluate_instruction_candidate(
        make_engine(),
        make_dataset(),
        lambda prompt, answer: {
            "instruction_following": 0.2,
            "safety": 0.5,
        },
        generation_config=GenerationConfig(max_new_tokens=1),
    )

    assert report["passed"] is False


def test_evaluator_cannot_claim_preference_without_preference_callback():
    report = evaluate_instruction_candidate(
        make_engine(),
        make_dataset(),
        passing_evaluator,
        generation_config=GenerationConfig(max_new_tokens=1),
    )

    assert report["candidate"]["preference"] is None
    assert report["scope"]["preference"] == "not_measured"
