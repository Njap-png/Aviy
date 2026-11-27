"""Synthetic-only tests for governed Aviy Base pretraining."""

from datetime import UTC, datetime

import pytest
import torch

from aviy.data.pipeline import DatasetRecord, PreparedDataset, prepare_dataset
from aviy.models.base_model import AviyBaseLM, BaseModelConfig
from aviy.tokenizer.bpe import ByteBPETokenizer
from aviy.training.authorization import (
    TrainingAuthorization,
    fingerprint_prepared_dataset,
)
from aviy.training.pretraining import pretrain_base_model


def make_dataset(language: str = "en") -> PreparedDataset:
    records = [
        DatasetRecord(
            text=f"This is a synthetic English training sentence number {index}.",
            source_id="synthetic-test-fixture",
            license="CC-BY-4.0",
            approved_for_training=True,
            source_uri="test://synthetic-pretraining",
            record_id=f"synthetic-{index}",
            language=language,
        )
        for index in range(40)
    ]
    return prepare_dataset(
        records,
        allowed_licenses=["CC-BY-4.0"],
        seed=42,
    )


def make_model() -> AviyBaseLM:
    return AviyBaseLM(
        BaseModelConfig(
            vocab_size=256,
            context_length=16,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            dropout=0.0,
        )
    )


def make_authorization(dataset: PreparedDataset) -> TrainingAuthorization:
    return TrainingAuthorization(
        stage="base_pretraining",
        dataset_fingerprint=fingerprint_prepared_dataset(dataset),
        approver="synthetic-test-only",
        authorized_at=datetime.now(UTC).isoformat(),
        evidence_reference="test://synthetic-fixture-only",
    )


def test_synthetic_pretraining_updates_model_and_saves_checkpoint(tmp_path):
    dataset = make_dataset()
    model = make_model()
    tokenizer = ByteBPETokenizer()
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    checkpoint = tmp_path / "base-pretraining.pt"

    report = pretrain_base_model(
        model,
        tokenizer,
        dataset,
        steps=2,
        batch_size=2,
        device_name="cpu",
        precision="fp32",
        checkpoint_path=checkpoint,
        authorization=make_authorization(dataset),
    )

    assert report["training_stage"] == "base_pretraining"
    assert report["global_step"] == 2
    assert report["random_initialization"] is True
    assert report["pretrained_weights_loaded"] is False
    assert report["capability_score_claimed"] is False
    assert checkpoint.is_file()
    assert checkpoint.with_suffix(".pt.sha256").is_file()
    assert all(torch.isfinite(torch.tensor(report["losses"])))
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )


def test_pretraining_rejects_missing_or_wrong_stage_authorization():
    dataset = make_dataset()
    with pytest.raises(PermissionError):
        pretrain_base_model(
            make_model(),
            ByteBPETokenizer(),
            dataset,
            steps=1,
            device_name="cpu",
        )

    wrong_stage = TrainingAuthorization(
        stage="supervised_finetuning",
        dataset_fingerprint=fingerprint_prepared_dataset(dataset),
        approver="synthetic-test-only",
        authorized_at=datetime.now(UTC).isoformat(),
        evidence_reference="test://wrong-stage",
    )
    with pytest.raises(PermissionError):
        pretrain_base_model(
            make_model(),
            ByteBPETokenizer(),
            dataset,
            steps=1,
            device_name="cpu",
            authorization=wrong_stage,
        )


def test_pretraining_rejects_non_english_metadata():
    dataset = make_dataset(language="sw")
    with pytest.raises(ValueError, match="English metadata"):
        pretrain_base_model(
            make_model(),
            ByteBPETokenizer(),
            dataset,
            steps=1,
            device_name="cpu",
            authorization=make_authorization(dataset),
        )


def test_pretraining_checkpoint_resume_and_configuration_binding(tmp_path):
    dataset = make_dataset()
    tokenizer = ByteBPETokenizer()
    checkpoint = tmp_path / "resume.pt"
    authorization = make_authorization(dataset)

    first = pretrain_base_model(
        make_model(),
        tokenizer,
        dataset,
        steps=2,
        batch_size=2,
        device_name="cpu",
        precision="fp32",
        checkpoint_path=checkpoint,
        authorization=authorization,
    )
    resumed_model = make_model()
    resumed = pretrain_base_model(
        resumed_model,
        tokenizer,
        dataset,
        steps=1,
        batch_size=2,
        device_name="cpu",
        precision="fp32",
        checkpoint_path=checkpoint,
        resume_from_checkpoint=checkpoint,
        authorization=authorization,
    )

    assert first["global_step"] == 2
    assert resumed["resumed_from_step"] == 2
    assert resumed["global_step"] == 3

    with pytest.raises(ValueError, match="does not match"):
        pretrain_base_model(
            make_model(),
            tokenizer,
            dataset,
            steps=1,
            batch_size=3,
            device_name="cpu",
            precision="fp32",
            checkpoint_path=checkpoint,
            resume_from_checkpoint=checkpoint,
            authorization=authorization,
        )



def test_document_windows_cover_all_next_token_pairs():
    from types import SimpleNamespace

    from aviy.training.pretraining import _PretrainingBatches

    tokenizer = ByteBPETokenizer()
    context = 4
    texts = ["abc", "abcdefgh", "abcdefghijklmnopqrst"]
    records = [SimpleNamespace(text=text) for text in texts]

    batches = _PretrainingBatches(
        records,
        tokenizer,
        batch_size=1,
        context_length=context,
    )

    expected_windows = sum(
        (len(tokenizer.encode(text)) - 1 + context - 1) // context
        for text in texts
    )
    assert len(batches._windows) == expected_windows

    for record_index, text in enumerate(texts):
        token_ids = tokenizer.encode(text)
        expected_pairs = list(zip(token_ids[:-1], token_ids[1:]))
        observed_pairs = []

        for window_index, (selected_record, offset) in enumerate(
            batches._windows
        ):
            if selected_record != record_index:
                continue
            batch_index = window_index
            inputs, targets = batches[batch_index]
            valid = targets[0] != -100
            observed_pairs.extend(
                zip(
                    inputs[0][valid].tolist(),
                    targets[0][valid].tolist(),
                )
            )

        assert observed_pairs == expected_pairs


def test_document_windows_do_not_cross_document_boundaries():
    from types import SimpleNamespace

    from aviy.training.pretraining import _PretrainingBatches

    tokenizer = ByteBPETokenizer()
    records = [
        SimpleNamespace(text="aaaa"),
        SimpleNamespace(text="zzzz"),
    ]
    batches = _PretrainingBatches(
        records,
        tokenizer,
        batch_size=1,
        context_length=16,
    )

    assert len(batches) == 2
    first_inputs, first_targets = batches[0]
    second_inputs, second_targets = batches[1]

    assert first_targets[0][first_targets[0] != -100].tolist() == (
        tokenizer.encode("aaaa")[1:]
    )
    assert second_targets[0][second_targets[0] != -100].tolist() == (
        tokenizer.encode("zzzz")[1:]
    )
    assert first_inputs.shape[0] == second_inputs.shape[0] == 1


def test_document_windows_pad_targets_with_ignore_index():
    from types import SimpleNamespace

    from aviy.training.pretraining import _PretrainingBatches

    tokenizer = ByteBPETokenizer()
    records = [
        SimpleNamespace(text="abcdefgh"),
        SimpleNamespace(text="xy"),
    ]
    batches = _PretrainingBatches(
        records,
        tokenizer,
        batch_size=2,
        context_length=16,
    )

    inputs, targets = batches[0]
    assert inputs.shape == targets.shape
    assert targets[1, 1:].eq(-100).all()
    assert targets[0].ne(-100).all()


def test_document_windows_reject_empty_or_single_token_documents():
    from types import SimpleNamespace

    from aviy.training.pretraining import _PretrainingBatches

    tokenizer = ByteBPETokenizer()
    with pytest.raises(ValueError, match="empty text"):
        _PretrainingBatches(
            [SimpleNamespace(text="   ")],
            tokenizer,
            batch_size=1,
            context_length=8,
        )
