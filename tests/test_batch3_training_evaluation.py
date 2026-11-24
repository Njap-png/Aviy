import torch
from torch import nn
import pytest

from aviy.training.engine import train_steps
from aviy.evaluation.benchmarks import (
    EVALUATION_DIMENSIONS,
    BenchmarkCase,
    BenchmarkSuite,
    run_benchmark,
)


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.output = nn.Linear(8, 16)

    def forward(self, token_ids):
        return self.output(self.embedding(token_ids))


def batches():
    inputs = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4], [2, 1, 0]], dtype=torch.long)
    return [(inputs, targets)]


def test_training_rejects_wrong_tensor_dtype():
    inputs = torch.ones((2, 3), dtype=torch.float32)
    targets = torch.ones((2, 3), dtype=torch.long)
    with pytest.raises(TypeError, match="torch.long"):
        train_steps(TinyLanguageModel(), [(inputs, targets)], steps=1, device_name="cpu")


def test_training_rejects_mismatched_batch_shapes():
    inputs = torch.ones((2, 3), dtype=torch.long)
    targets = torch.ones((2, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="matching"):
        train_steps(TinyLanguageModel(), [(inputs, targets)], steps=1, device_name="cpu")


def test_training_rejects_negative_token_ids():
    inputs = torch.tensor([[-1, 2]], dtype=torch.long)
    targets = torch.tensor([[1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="negative"):
        train_steps(TinyLanguageModel(), [(inputs, targets)], steps=1, device_name="cpu")


def test_scheduler_state_survives_checkpoint_resume(tmp_path):
    checkpoint = tmp_path / "scheduled.pt"
    model = TinyLanguageModel()
    first = train_steps(
        model, batches(), steps=2, device_name="cpu",
        checkpoint_path=checkpoint,
        scheduler_total_steps=4,
        scheduler_warmup_steps=1,
        minimum_learning_rate_ratio=0.1,
    )
    resumed = train_steps(
        model, batches(), steps=2, device_name="cpu",
        resume_from_checkpoint=checkpoint,
        checkpoint_path=checkpoint,
        scheduler_total_steps=4,
        scheduler_warmup_steps=1,
        minimum_learning_rate_ratio=0.1,
    )
    assert first["global_step"] == 2
    assert resumed["resumed_from_step"] == 2
    assert resumed["global_step"] == 4
    assert resumed["scheduler_enabled"] is True
    assert resumed["final_learning_rate"] <= first["final_learning_rate"]


def test_scheduler_options_require_total_steps():
    with pytest.raises(ValueError, match="require scheduler_total_steps"):
        train_steps(
            TinyLanguageModel(), batches(), steps=1, device_name="cpu",
            scheduler_warmup_steps=1,
        )


def test_benchmark_reports_repeatability_separately_from_accuracy():
    suite = BenchmarkSuite(
        "repeatability",
        tuple(
            BenchmarkCase(f"id-{i}", dimension, f"prompt {i}", f"answer {i}")
            for i, dimension in enumerate(EVALUATION_DIMENSIONS)
        ),
    )
    calls = {}

    def predictor(prompt):
        count = calls.get(prompt, 0)
        calls[prompt] = count + 1
        return "stable" if count % 2 == 0 else "different"

    report = run_benchmark(suite, predictor, repeatability_runs=2)
    assert report["accuracy"] == 0.0
    assert report["repeatability"] == 0.0
    assert report["repeatability_runs"] == 2
    assert report["metric_note"].find("repeatability") >= 0


def test_benchmark_rejects_invalid_repeatability_count():
    suite = BenchmarkSuite(
        "one",
        tuple(
            BenchmarkCase(f"id-{i}", dimension, f"prompt {i}", f"answer {i}")
            for i, dimension in enumerate(EVALUATION_DIMENSIONS)
        ),
    )
    with pytest.raises(ValueError, match="repeatability_runs"):
        run_benchmark(suite, lambda _: "answer", repeatability_runs=0)
