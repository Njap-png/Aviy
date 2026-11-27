from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import pytest
import torch

from aviy.cognition.reasoning import (
    Planner,
    PlanStep,
    detect_reasoning_failure,
    evaluate_reasoning,
)
from aviy.evaluation.benchmarks import (
    EVALUATION_DIMENSIONS,
    BenchmarkCase,
    BenchmarkSuite,
    check_contamination,
    compare_regression,
    run_benchmark,
)
from aviy.inference.generation import (
    GenerationCache,
    GenerationConfig,
    benchmark_generation,
    generate_tokens,
)
from aviy.knowledge.retrieval import InMemoryGraph, InMemoryVectorIndex
from aviy.learning import instruction as instruction_module
from aviy.learning.instruction import (
    InstructionRecord,
    evaluate_alignment,
    prepare_instruction_dataset,
    train_supervised_finetuning,
)
from aviy.memory.store import MemoryStore
from aviy.models.base_model import AviyBaseLM, BaseModelConfig
from aviy.tokenizer.bpe import ByteBPETokenizer
from aviy.training import checkpoints as checkpoint_module
from aviy.training.authorization import (
    TrainingAuthorization,
    fingerprint_prepared_dataset,
)


def make_suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        "all-dimensions",
        tuple(
            BenchmarkCase(
                f"case-{index}", dimension, f"prompt {index}", f"answer {index}"
            )
            for index, dimension in enumerate(EVALUATION_DIMENSIONS)
        ),
    )


def test_benchmark_covers_all_dimensions_and_is_reproducible() -> None:
    suite = make_suite()
    predictor = lambda prompt: "answer " + prompt.split()[-1]
    first = run_benchmark(suite, predictor, seed=42)
    second = run_benchmark(suite, predictor, seed=42)
    assert first["accuracy"] == 1.0
    dimensions = first["dimensions"]
    assert isinstance(dimensions, dict)
    assert set(dimensions.keys()) == set(EVALUATION_DIMENSIONS)
    assert first["reproducibility_fingerprint"] == second["reproducibility_fingerprint"]


def test_benchmark_rejects_missing_dimension() -> None:
    suite = BenchmarkSuite(
        "partial", (BenchmarkCase("a", "reasoning", "prompt", "answer"),)
    )
    with pytest.raises(ValueError, match="missing dimensions"):
        run_benchmark(suite, lambda _: "answer")


def test_benchmark_contamination_and_regression() -> None:
    suite = make_suite()
    result = check_contamination(suite, ["  PROMPT 0 "])
    assert not result["passed"]
    assert result["overlapping_case_ids"] == ["case-0"]
    assert compare_regression({"accuracy": 0.9}, {"accuracy": 0.89}, maximum_drop=0.02)[
        "passed"
    ]
    assert not compare_regression(
        {"accuracy": 0.9}, {"accuracy": 0.8}, maximum_drop=0.02
    )["passed"]


def test_memory_owner_access_versioning_expiry_and_recovery() -> None:
    store = MemoryStore()
    first = store.put(
        record_id="m1",
        owner_id="owner-a",
        kind="episodic",
        content="first event",
        provenance="conversation:1",
    )
    second = store.put(
        record_id="m1",
        owner_id="owner-a",
        kind="episodic",
        content="updated event",
        provenance="conversation:2",
        conflict="replace",
    )
    assert first.version == 1 and second.version == 2
    with pytest.raises(PermissionError):
        store.get("m1", "owner-b")
    snapshot = store.snapshot()
    restored = MemoryStore()
    restored.restore(snapshot)
    restored_record = restored.get("m1", "owner-a")
    assert restored_record is not None
    assert restored_record.content == "updated event"
    assert restored_record.version == 2


def test_memory_rejects_missing_provenance_and_handles_retention() -> None:
    store = MemoryStore()
    with pytest.raises(ValueError):
        store.put(
            record_id="m",
            owner_id="o",
            kind="working",
            content="text",
            provenance="",
        )
    store.put(
        record_id="temporary",
        owner_id="o",
        kind="working",
        content="short-lived",
        provenance="test",
        ttl_seconds=10,
    )
    record = store.get("temporary", "o")
    assert record is not None
    assert record.expires_at is not None
    assert store.expire(record.expires_at + 1) == 1


def test_graph_and_vector_retrieval() -> None:
    graph = InMemoryGraph()
    graph.add_node("a", {"type": "concept"})
    graph.add_node("b", {"type": "concept"})
    graph.add_edge("a", "b", "related")
    assert graph.related("a", "related") == ["b"]

    index = InMemoryVectorIndex()
    index.upsert("x", [1.0, 0.0])
    index.upsert("y", [0.0, 1.0])
    assert index.query([0.9, 0.1])[0][0] == "x"


def test_planner_orders_dependencies_and_detects_cycles() -> None:
    planner = Planner()
    result = planner.plan(
        (
            PlanStep("second", "Second", ("first",)),
            PlanStep("first", "First"),
        )
    )
    assert [step.step_id for step in result] == ["first", "second"]
    with pytest.raises(ValueError, match="cycle"):
        planner.plan(
            (
                PlanStep("a", "A", ("b",)),
                PlanStep("b", "B", ("a",)),
            )
        )


def test_reasoning_verification_uncertainty_and_failure_detection() -> None:
    result = evaluate_reasoning(
        (PlanStep("a", "Check"), PlanStep("b", "Calculate")),
        "answer",
        lambda step: step.step_id == "a",
        confidence=0.8,
        expected_correctness=False,
    )
    assert result.failed_step_ids == ("b",)
    assert result.brier_score == pytest.approx(0.64)
    assert not detect_reasoning_failure(result)["passed"]


def make_model() -> AviyBaseLM:
    return AviyBaseLM(
        BaseModelConfig(
            vocab_size=256,
            context_length=32,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
        )
    )


def test_generation_is_deterministic_and_context_bounded() -> None:
    model = make_model()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    config = GenerationConfig(max_new_tokens=4, do_sample=True, seed=7)
    first = generate_tokens(model, prompt, config)
    second = generate_tokens(model, prompt, config)
    assert torch.equal(first, second)
    assert first.shape == (1, 7)


def test_generation_cache_is_bounded_and_lru() -> None:
    cache = GenerationCache(maximum_entries=2)
    cache.put(("a",), (1,))
    cache.put(("b",), (2,))
    assert cache.get(("a",)) == (1,)
    cache.put(("c",), (3,))
    assert cache.get(("b",)) is None
    assert len(cache) == 2


def test_generation_benchmark_reports_latency() -> None:
    model = make_model()
    report = benchmark_generation(
        model,
        torch.tensor([[1, 2]]),
        GenerationConfig(max_new_tokens=2),
        repeats=1,
    )
    assert report["generated_tokens"] == 2
    assert report["mean_latency_seconds"] >= 0


def make_instruction_dataset():
    records = [
        InstructionRecord(
            prompt=f"Question {index}",
            response=f"Answer {index}",
            source_id="approved-source",
            license="CC-BY-4.0",
            approved_for_training=True,
            source_uri="https://example.org/dataset",
            record_id=f"record-{index}",
        )
        for index in range(30)
    ]
    return prepare_instruction_dataset(records, allowed_licenses=["CC-BY-4.0"], seed=42)


def sft_test_authorization(dataset) -> TrainingAuthorization:
    return TrainingAuthorization(
        stage="supervised_finetuning",
        dataset_fingerprint=fingerprint_prepared_dataset(dataset),
        approver="synthetic-test",
        authorized_at=datetime.now(UTC).isoformat(),
        evidence_reference="test://synthetic-sft-fixture",
    )


def test_instruction_preparation_uses_approved_governed_data() -> None:
    dataset = make_instruction_dataset()
    assert dataset.audit.passed
    assert dataset.split("train")
    assert dataset.split("validation")
    assert dataset.split("test")


@pytest.mark.parametrize("authorization_kind", ["missing", "wrong_stage", "stale_dataset"])
def test_instruction_sft_rejects_missing_or_mismatched_authorization(
    authorization_kind: str,
) -> None:
    dataset = make_instruction_dataset()
    valid = sft_test_authorization(dataset)

    if authorization_kind == "missing":
        authorization = None
    elif authorization_kind == "wrong_stage":
        authorization = TrainingAuthorization(
            stage="tokenizer_training",
            dataset_fingerprint=valid.dataset_fingerprint,
            approver="synthetic-test",
            authorized_at=datetime.now(UTC).isoformat(),
            evidence_reference="test://wrong-stage",
        )
    else:
        authorization = TrainingAuthorization(
            stage="supervised_finetuning",
            dataset_fingerprint="0" * 64,
            approver="synthetic-test",
            authorized_at=datetime.now(UTC).isoformat(),
            evidence_reference="test://stale-dataset",
        )

    with pytest.raises(PermissionError):
        train_supervised_finetuning(
            make_model(),
            ByteBPETokenizer(),
            dataset,
            authorization=authorization,
        )


def test_instruction_sft_updates_model_using_training_split_only() -> None:
    dataset = make_instruction_dataset()
    tokenizer = ByteBPETokenizer.train(
        [f"Question {i} Answer {i}" for i in range(30)],
        vocabulary_size=256,
    )
    model = make_model()
    before = model.token_embedding.weight.detach().clone()
    report = train_supervised_finetuning(
        model,
        tokenizer,
        dataset,
        epochs=1,
        maximum_length=32,

        authorization=sft_test_authorization(dataset),
    )
    optimization_steps = report["optimization_steps"]
    assert isinstance(optimization_steps, int)
    assert optimization_steps > 0
    assert not torch.equal(before, model.token_embedding.weight)


def test_alignment_evaluation_checks_instruction_and_safety() -> None:
    report = evaluate_alignment(
        [object(), object()],
        lambda _: {
            "instruction_following": 0.9,
            "safety": 1.0,
            "preference": 0.8,
        },
    )
    assert report.get("passed") is True
    preference = report["preference"]
    assert isinstance(preference, (int, float))
    assert preference == pytest.approx(0.8)


def test_instruction_dataset_rejects_unapproved_source() -> None:
    bad = InstructionRecord(
        prompt="Question",
        response="Answer",
        source_id="source",
        license="Unknown",
        approved_for_training=False,
        source_uri="https://example.org/data",
        record_id="bad",
    )
    dataset = prepare_instruction_dataset([bad], allowed_licenses=["CC-BY-4.0"])
    assert dataset.audit.passed is False
    with pytest.raises(ValueError, match="audit"):
        train_supervised_finetuning(make_model(), ByteBPETokenizer(), dataset)

def test_instruction_sft_reports_response_only_and_validation_losses() -> None:
    dataset = make_instruction_dataset()
    tokenizer = ByteBPETokenizer.train(
        [f"Question {i} Answer {i}" for i in range(30)],
        vocabulary_size=256,
    )
    model = make_model()

    report = train_supervised_finetuning(
        model,
        tokenizer,
        dataset,
        epochs=2,
        maximum_length=32,
        batch_size=4,
        seed=42,

        authorization=sft_test_authorization(dataset),
    )

    assert report["response_only_loss"] is True
    assert report["completed_epochs"] == 2
    assert cast(int, report["optimization_steps"]) > 0
    validation_losses = cast(list[float], report["validation_losses"])
    assert len(validation_losses) == 2
    assert all(torch.isfinite(torch.tensor(value)) for value in validation_losses)
    assert cast(int, report["response_tokens_seen"]) > 0


def test_instruction_sft_checkpoint_resumes_at_next_epoch(tmp_path) -> None:
    dataset = make_instruction_dataset()
    tokenizer = ByteBPETokenizer.train(
        [f"Question {i} Answer {i}" for i in range(30)],
        vocabulary_size=256,
    )
    checkpoint = tmp_path / "instruction_sft.pt"

    first_model = make_model()
    first = train_supervised_finetuning(
        first_model,
        tokenizer,
        dataset,
        epochs=1,
        maximum_length=32,
        batch_size=4,
        seed=42,
        checkpoint_path=checkpoint,

        authorization=sft_test_authorization(dataset),
    )

    assert first["completed_epochs"] == 1
    assert checkpoint.is_file()
    assert checkpoint.with_suffix(".pt.sha256").is_file()

    resumed_model = make_model()
    resumed = train_supervised_finetuning(
        resumed_model,
        tokenizer,
        dataset,
        epochs=2,
        maximum_length=32,
        batch_size=4,
        seed=42,
        checkpoint_path=checkpoint,
        resume_from_checkpoint=checkpoint,

        authorization=sft_test_authorization(dataset),
    )

    assert resumed["resumed_from_epoch"] == 1
    assert resumed["completed_epochs"] == 2
    assert cast(int, resumed["optimization_steps"]) > 0
    assert cast(int, resumed["global_step"]) > cast(int, first["global_step"])
    assert cast(int, resumed["response_tokens_seen"]) > cast(
        int, first["response_tokens_seen"]
    )


@pytest.mark.parametrize(
    "corruption, expected_error",
    [
        ("step", "optimization step count"),
        ("missing_response_tokens", "response-token count"),
        ("zero_response_tokens", "response-token count"),
        ("boolean_response_tokens", "response-token count"),
    ],
)
def test_instruction_sft_rejects_corrupt_resume_metadata_before_restore(
    tmp_path, monkeypatch, corruption: str, expected_error: str
) -> None:
    dataset = make_instruction_dataset()
    tokenizer = ByteBPETokenizer.train(
        [f"Question {i} Answer {i}" for i in range(30)],
        vocabulary_size=256,
    )
    checkpoint = tmp_path / "instruction_sft_corrupt.pt"

    train_supervised_finetuning(
        make_model(),
        tokenizer,
        dataset,
        epochs=1,
        maximum_length=32,
        batch_size=4,
        seed=42,
        checkpoint_path=checkpoint,

        authorization=sft_test_authorization(dataset),
    )

    payload = checkpoint_module.load_verified_checkpoint(
        checkpoint, map_location="cpu"
    )
    sft_metadata = payload["metadata"]["sft"]

    if corruption == "step":
        payload["step"] += 1
    elif corruption == "missing_response_tokens":
        sft_metadata.pop("response_tokens_seen")
    elif corruption == "zero_response_tokens":
        sft_metadata["response_tokens_seen"] = 0
    elif corruption == "boolean_response_tokens":
        sft_metadata["response_tokens_seen"] = True

    monkeypatch.setattr(
        checkpoint_module,
        "load_verified_checkpoint",
        lambda *args, **kwargs: payload,
    )

    def fail_if_restored(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid checkpoint reached model/optimizer restoration")

    monkeypatch.setattr(
        instruction_module, "load_checkpoint", fail_if_restored
    )

    with pytest.raises(ValueError, match=expected_error):
        train_supervised_finetuning(
            make_model(),
            tokenizer,
            dataset,
            epochs=2,
            maximum_length=32,
            batch_size=4,
            seed=42,
            checkpoint_path=checkpoint,
            resume_from_checkpoint=checkpoint,

            authorization=sft_test_authorization(dataset),
        )


def test_instruction_sft_rejects_incompatible_resume_settings(tmp_path) -> None:
    dataset = make_instruction_dataset()
    tokenizer = ByteBPETokenizer.train(
        [f"Question {i} Answer {i}" for i in range(30)],
        vocabulary_size=256,
    )
    checkpoint = tmp_path / "instruction_sft.pt"

    train_supervised_finetuning(
        make_model(),
        tokenizer,
        dataset,
        epochs=1,
        maximum_length=32,
        batch_size=4,
        seed=42,
        checkpoint_path=checkpoint,

        authorization=sft_test_authorization(dataset),
    )

    with pytest.raises(ValueError, match="incompatible"):
        train_supervised_finetuning(
            make_model(),
            tokenizer,
            dataset,
            epochs=2,
            maximum_length=32,
            batch_size=8,
            seed=42,
            checkpoint_path=checkpoint,
            resume_from_checkpoint=checkpoint,

            authorization=sft_test_authorization(dataset),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"maximum_length": 1},
        {"seed": -1},
        {"learning_rate": float("nan")},
    ],
)
def test_instruction_sft_rejects_invalid_training_settings(kwargs) -> None:
    dataset = make_instruction_dataset()
    tokenizer = ByteBPETokenizer()
    with pytest.raises(ValueError):
        train_supervised_finetuning(
            make_model(),
            tokenizer,
            dataset,
            **kwargs,
        )
