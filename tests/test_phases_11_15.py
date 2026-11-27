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


@pytest.mark.parametrize("do_sample", [False, True])
def test_generation_masks_logits_outside_allowed_vocabulary(monkeypatch, do_sample):
    model = make_model()

    def fixed_forward(input_ids):
        logits = torch.zeros(
            (*input_ids.shape, 256), device=input_ids.device
        )
        logits[..., 200] = 1000.0
        logits[..., 7] = 10.0
        return logits

    monkeypatch.setattr(model, "forward", fixed_forward)

    output = generate_tokens(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        GenerationConfig(max_new_tokens=8, do_sample=do_sample, seed=42),
        allowed_vocab_size=128,
    )

    generated_ids = output[0, 2:].tolist()
    assert len(generated_ids) == 8
    assert all(0 <= token_id < 128 for token_id in generated_ids)


@pytest.mark.parametrize("invalid_limit", [0, -1, 257, True])
def test_generation_rejects_invalid_allowed_vocabulary_limit(invalid_limit):
    with pytest.raises(ValueError, match="allowed_vocab_size"):
        generate_tokens(
            make_model(),
            torch.tensor([[1, 2]], dtype=torch.long),
            GenerationConfig(max_new_tokens=1),
            allowed_vocab_size=invalid_limit,
        )


def test_generation_rejects_eos_token_outside_allowed_vocabulary():
    with pytest.raises(ValueError, match="eos_token_id"):
        generate_tokens(
            make_model(),
            torch.tensor([[1, 2]], dtype=torch.long),
            GenerationConfig(max_new_tokens=1, eos_token_id=200),
            allowed_vocab_size=128,
        )


def test_generation_rejects_prompt_ids_outside_allowed_vocabulary():
    with pytest.raises(ValueError, match="outside the allowed vocabulary"):
        generate_tokens(
            make_model(),
            torch.tensor([[1, 200]], dtype=torch.long),
            GenerationConfig(max_new_tokens=1),
            allowed_vocab_size=128,
        )


# Synthetic generation reliability regression tests.
GENERATION_RELIABILITY_TEST_MARKER = "generation-reliability-v1"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": True},
        {"max_new_tokens": -1},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": 0.0},
        {"top_k": True},
        {"top_k": -1},
        {"top_p": float("nan")},
        {"top_p": float("inf")},
        {"top_p": 0.0},
        {"do_sample": 1},
        {"eos_token_id": -1},
        {"eos_token_id": True},
        {"seed": True},
        {"seed": -1},
    ],
)
def test_generation_rejects_invalid_configuration_values(kwargs):
    expected_exception = TypeError if "do_sample" in kwargs else ValueError
    with pytest.raises(expected_exception):
        GenerationConfig(**kwargs).validate()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_generation_rejects_non_finite_model_logits(monkeypatch, bad_value):
    model = make_model()

    def invalid_forward(input_ids):
        logits = torch.zeros((*input_ids.shape, 256))
        logits[..., 7] = bad_value
        return logits

    monkeypatch.setattr(model, "forward", invalid_forward)

    with pytest.raises(ValueError, match="non-finite"):
        generate_tokens(
            model,
            torch.tensor([[1, 2]], dtype=torch.long),
            GenerationConfig(max_new_tokens=1),
        )


def test_generation_rejects_malformed_model_logits(monkeypatch):
    model = make_model()

    def malformed_forward(input_ids):
        return torch.zeros((1, 1, 256))

    monkeypatch.setattr(model, "forward", malformed_forward)

    with pytest.raises(ValueError, match="logits must have shape"):
        generate_tokens(
            model,
            torch.tensor([[1, 2]], dtype=torch.long),
            GenerationConfig(max_new_tokens=1),
        )


def test_generation_rejects_non_tensor_and_non_integer_prompts():
    model = make_model()

    with pytest.raises(TypeError, match="torch.Tensor"):
        generate_tokens(
            model,
            cast(torch.Tensor, [[1, 2]]),
            GenerationConfig(max_new_tokens=1),
        )

    with pytest.raises(TypeError, match="integer token IDs"):
        generate_tokens(
            model,
            torch.tensor([[1.0, 2.0]]),
            GenerationConfig(max_new_tokens=1),
        )


def test_generation_stops_immediately_on_eos_and_restores_training_mode(
    monkeypatch,
):
    model = make_model()
    model.train()

    def eos_forward(input_ids):
        logits = torch.zeros((*input_ids.shape, 256))
        logits[..., 9] = 100.0
        return logits

    monkeypatch.setattr(model, "forward", eos_forward)

    output = generate_tokens(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        GenerationConfig(max_new_tokens=10, eos_token_id=9),
    )

    assert output.tolist() == [[1, 2, 9]]
    assert model.training is True


def test_generation_preserves_evaluation_mode(monkeypatch):
    model = make_model().eval()

    def fixed_forward(input_ids):
        logits = torch.zeros((*input_ids.shape, 256))
        logits[..., 5] = 10.0
        return logits

    monkeypatch.setattr(model, "forward", fixed_forward)

    output = generate_tokens(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        GenerationConfig(max_new_tokens=3),
    )

    assert output.shape == (1, 5)
    assert model.training is False


def test_generation_respects_context_window(monkeypatch):
    model = make_model()
    observed_lengths = []

    def fixed_forward(input_ids):
        observed_lengths.append(input_ids.shape[1])
        return torch.zeros((*input_ids.shape, 256))

    monkeypatch.setattr(model, "forward", fixed_forward)

    prompt = torch.tensor([list(range(40))], dtype=torch.long)
    generate_tokens(
        model,
        prompt,
        GenerationConfig(max_new_tokens=2),
    )

    assert observed_lengths == [32, 32]


def test_generation_top_k_sampling_stays_within_selected_candidates(
    monkeypatch,
):
    model = make_model()

    def controlled_forward(input_ids):
        logits = torch.full((*input_ids.shape, 256), -100.0)
        logits[..., 10] = 10.0
        logits[..., 11] = 9.0
        logits[..., 12] = 8.0
        return logits

    monkeypatch.setattr(model, "forward", controlled_forward)

    output = generate_tokens(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        GenerationConfig(
            max_new_tokens=32,
            do_sample=True,
            top_k=2,
            seed=42,
        ),
    )

    assert set(output[0, 2:].tolist()).issubset({10, 11})


def test_generation_top_p_sampling_keeps_probability_mass_candidates(
    monkeypatch,
):
    model = make_model()

    def controlled_forward(input_ids):
        logits = torch.full((*input_ids.shape, 256), -100.0)
        logits[..., 10] = 10.0
        logits[..., 11] = 9.0
        logits[..., 12] = 0.0
        return logits

    monkeypatch.setattr(model, "forward", controlled_forward)

    output = generate_tokens(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        GenerationConfig(
            max_new_tokens=32,
            do_sample=True,
            top_p=0.5,
            seed=42,
        ),
    )

    assert set(output[0, 2:].tolist()).issubset({10, 11})


def test_generation_zero_new_tokens_returns_prompt_unchanged():
    model = make_model()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    output = generate_tokens(
        model,
        prompt,
        GenerationConfig(max_new_tokens=0),
        allowed_vocab_size=256,
    )

    assert torch.equal(output, prompt)


# Synthetic cache and benchmark hardening regression tests.
GENERATION_CACHE_BENCHMARK_MARKER = "generation-cache-benchmark-v1"


@pytest.mark.parametrize("maximum_entries", [0, -1, True, 1.5])
def test_generation_cache_rejects_invalid_capacity(maximum_entries):
    with pytest.raises(ValueError, match="positive integer"):
        GenerationCache(maximum_entries=maximum_entries)


def test_generation_cache_validates_keys_and_values():
    cache = GenerationCache()

    with pytest.raises(TypeError, match="key must be a tuple"):
        cache.put(["prompt"], (1, 2))

    with pytest.raises(TypeError, match="hashable"):
        cache.put(("prompt", []), (1, 2))

    with pytest.raises(TypeError, match="value must be a tuple"):
        cache.put(("prompt",), [1, 2])

    with pytest.raises(ValueError, match="non-negative integer"):
        cache.put(("prompt",), (1, -2))

    with pytest.raises(ValueError, match="non-negative integer"):
        cache.put(("prompt",), (True,))


def test_generation_cache_isolates_keys_and_replaces_existing_values():
    cache = GenerationCache(maximum_entries=2)

    cache.put(("prompt-a", "greedy"), (1, 2))
    cache.put(("prompt-a", "sampled"), (3, 4))
    cache.put(("prompt-a", "greedy"), (5, 6))

    assert cache.get(("prompt-a", "greedy")) == (5, 6)
    assert cache.get(("prompt-a", "sampled")) == (3, 4)
    assert len(cache) == 2


def test_generation_cache_clear_removes_all_entries():
    cache = GenerationCache(maximum_entries=3)
    cache.put(("a",), (1,))
    cache.put(("b",), (2,))

    cache.clear()

    assert len(cache) == 0
    assert cache.get(("a",)) is None
    assert cache.get(("b",)) is None


def test_generation_cache_get_rejects_unhashable_key():
    cache = GenerationCache()

    with pytest.raises(TypeError, match="hashable"):
        cache.get(("prompt", []))


@pytest.mark.parametrize("repeats", [0, -1, True, 1.5])
def test_generation_benchmark_rejects_invalid_repeat_counts(repeats):
    model = make_model()

    with pytest.raises(ValueError, match="repeats must be a positive integer"):
        benchmark_generation(
            model,
            torch.tensor([[1, 2]], dtype=torch.long),
            GenerationConfig(max_new_tokens=1),
            repeats=repeats,
        )


def test_generation_benchmark_reports_consistent_repeat_metrics():
    model = make_model()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    report = benchmark_generation(
        model,
        prompt,
        GenerationConfig(max_new_tokens=3, eos_token_id=None, seed=17),
        repeats=3,
    )

    assert report["repeats"] == 3
    assert report["generated_tokens"] == 3
    assert report["mean_latency_seconds"] >= 0
    assert report["tokens_per_second"] > 0


def test_generation_benchmark_validates_generation_configuration():
    model = make_model()

    with pytest.raises(ValueError, match="temperature"):
        benchmark_generation(
            model,
            torch.tensor([[1, 2]], dtype=torch.long),
            GenerationConfig(temperature=float("nan")),
            repeats=1,
        )
