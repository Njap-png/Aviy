from dataclasses import replace

from aviy.learning.instruction import (
    InstructionRecord,
    _dataset_fingerprint,
    prepare_instruction_dataset,
)


def test_instruction_preparation_preserves_extended_provenance():
    record = InstructionRecord(
        prompt="Explain a basic networking concept.",
        response="A network connects devices so they can communicate.",
        source_id="approved-instruction-source-v1",
        license="CC-BY-4.0",
        approved_for_training=True,
        source_uri="https://example.org/dataset/records",
        record_id="instruction-provenance-001",
        dataset_revision="0123456789abcdef",
        license_uri="https://creativecommons.org/licenses/by/4.0/",
        language="en",
        language_script="Latn",
        retrieved_at="2026-11-25T10:00:00Z",
        training_stage="supervised_finetuning",
        content_category="technical_instruction",
        processing_history=("source_retrieved",),
    )

    prepared = prepare_instruction_dataset(
        [record],
        allowed_licenses=["CC-BY-4.0"],
    )

    assert prepared.audit.passed
    assert prepared.audit.accepted == 1
    assert len(prepared.records) == 1

    result = prepared.records[0]

    assert result.dataset_revision == "0123456789abcdef"
    assert result.license_uri == "https://creativecommons.org/licenses/by/4.0/"
    assert result.language == "en"
    assert result.language_script == "Latn"
    assert result.retrieved_at == "2026-11-25T10:00:00Z"
    assert result.training_stage == "supervised_finetuning"
    assert result.content_category == "technical_instruction"
    assert result.processing_history == (
        "source_retrieved",
        "unicode_nfc_whitespace_normalization",
    )
    assert result.privacy_screening == "automated_checks_passed"
    assert result.duplicate_screening == "automated_checks_passed"

    assert prepared.split(result.split) == (result,)


def test_instruction_dataset_fingerprint_includes_all_provenance():
    record = InstructionRecord(
        prompt="Explain a basic networking concept.",
        response="A network connects devices.",
        source_id="approved-instruction-source-v1",
        license="CC-BY-4.0",
        approved_for_training=True,
        source_uri="https://example.org/dataset/records",
        record_id="instruction-fingerprint-001",
        dataset_revision="revision-a",
        license_uri="https://creativecommons.org/licenses/by/4.0/",
        language="en",
        language_script="Latn",
        retrieved_at="2026-11-25T10:00:00Z",
        training_stage="supervised_finetuning",
        content_category="technical_instruction",
        processing_history=("source_retrieved",),
    )

    prepared = prepare_instruction_dataset(
        [record],
        allowed_licenses=["CC-BY-4.0"],
    )
    original_hash = _dataset_fingerprint(prepared)

    changes = {
        "dataset_revision": "revision-b",
        "license_uri": "https://example.org/other-license",
        "language": "sw",
        "language_script": "Cyrl",
        "retrieved_at": "2026-11-26T10:00:00Z",
        "training_stage": "continued_pretraining",
        "content_category": "mathematics",
        "processing_history": ("source_retrieved", "additional_review"),
        "privacy_screening": "reviewed",
        "duplicate_screening": "reviewed",
    }

    for field, changed_value in changes.items():
        changed_record = replace(prepared.records[0], **{field: changed_value})
        changed_dataset = replace(prepared, records=(changed_record,))
        assert _dataset_fingerprint(changed_dataset) != original_hash, field
