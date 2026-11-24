from __future__ import annotations

import json

import pytest
import torch

from aviy.inference.generation import GenerationConfig
from aviy.inference.runtime import InferenceEngine, save_inference_artifact
from aviy.models.base_model import AviyBaseLM, BaseModelConfig
from aviy.tokenizer.bpe import ByteBPETokenizer


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


def make_tokenizer() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(
        ["Aviy inference test text", "A small test corpus for Aviy"],
        vocabulary_size=256,
    )


def test_runtime_generates_text_and_restores_training_mode():
    model = make_model()
    tokenizer = make_tokenizer()
    engine = InferenceEngine(model, tokenizer)
    model.train()

    output = engine.generate(
        "Aviy",
        GenerationConfig(max_new_tokens=3, do_sample=False),
    )

    assert isinstance(output, str)
    assert model.training is True


def test_runtime_rejects_empty_and_overlong_prompts():
    engine = InferenceEngine(make_model(), make_tokenizer())

    with pytest.raises(ValueError, match="at least one token"):
        engine.generate("")

    with pytest.raises(ValueError, match="context length"):
        engine.generate("x" * 33)


def test_artifact_round_trip_preserves_model_outputs(tmp_path):
    model = make_model().eval()
    tokenizer = make_tokenizer()
    sample = torch.tensor([[65, 66, 67]], dtype=torch.long)

    with torch.no_grad():
        expected = model(sample)

    artifact = tmp_path / "artifact"
    save_inference_artifact(model, tokenizer, artifact)
    restored = InferenceEngine.from_artifact(artifact)

    with torch.no_grad():
        actual = restored.model(sample)

    torch.testing.assert_close(actual, expected)
    assert restored.tokenizer.encode("Aviy") == tokenizer.encode("Aviy")
    assert restored.inspect()["pretrained_weights"] is False


def test_artifact_export_refuses_to_overwrite(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    with pytest.raises(FileExistsError):
        save_inference_artifact(make_model(), make_tokenizer(), artifact)


def test_artifact_loader_detects_modified_weights(tmp_path):
    artifact = tmp_path / "artifact"
    save_inference_artifact(make_model(), make_tokenizer(), artifact)

    weights = artifact / "model.pt"
    weights.write_bytes(weights.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="integrity check"):
        InferenceEngine.from_artifact(artifact)


def test_artifact_loader_rejects_unsupported_manifest(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(
        json.dumps({"format": "unknown"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported"):
        InferenceEngine.from_artifact(artifact)


def test_runtime_checks_tokenizer_vocabulary_compatibility():
    model = make_model()
    tokenizer = ByteBPETokenizer()
    assert tokenizer.vocabulary_size == 256
    InferenceEngine(model, tokenizer)

    too_large = ByteBPETokenizer.train(
        ["repeated repeated repeated repeated"],
        vocabulary_size=300,
        minimum_pair_frequency=1,
    )
    small_model = AviyBaseLM(
        BaseModelConfig(
            vocab_size=256,
            context_length=32,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
        )
    )
    with pytest.raises(ValueError, match="does not fit"):
        InferenceEngine(small_model, too_large)
