"""Tests for the strict dataset-pipeline governance adapter."""

from dataclasses import replace
from typing import Any

import pytest

from aviy.corpus_governance.engine import (
    GovernanceError,
    SourceRegistration,
    SourceRegistry,
)
from aviy.corpus_governance.pipeline_adapter import (
    HumanReviewDecision,
    PrivacyReviewDecision,
    RecordGovernanceMetadata,
    govern_prepared_dataset,
)
from aviy.data.pipeline import DatasetAudit, DatasetRecord, PreparedDataset

SOURCE_URI = "https://example.org/datasets/aviy-v1"


def approved_registry():
    return SourceRegistry([
        SourceRegistration(
            source_id="approved-source",
            source_name="Approved Example Dataset",
            source_uri=SOURCE_URI,
            revision="revision-1",
            license_id="CC-BY-4.0",
            license_status="approved",
            approval_status="approved",
            training_allowed=True,
            attribution_required=True,
            attribution_text="Source: Approved Example Dataset",
            reviewer="source-reviewer",
            approved_at="2026-11-26T08:00:00Z",
            rights_evidence=synthetic_rights_evidence(),
        )
    ])


def pipeline_record(record_id="record-1", **changes):
    values = {
        "text": "This is a sufficiently long example for governance testing.",
        "source_id": "approved-source",
        "license": "CC-BY-4.0",
        "approved_for_training": True,
        "source_uri": SOURCE_URI,
        "record_id": record_id,
        "dataset_revision": "revision-1",
        "language": "en",
        "language_script": "Latn",
        "content_category": "general_language",
    }
    values.update(changes)
    return DatasetRecord(**values)


def prepared_dataset(*records):
    audit = DatasetAudit(
        accepted=len(records),
        rejected=0,
        duplicates_removed=0,
        split_counts={"train": len(records), "validation": 0, "test": 0},
        provenance_complete=True,
        licenses_approved=True,
        privacy_scan_passed=True,
        split_integrity_passed=True,
        seed=42,
    )
    return PreparedDataset(records=tuple(records), audit=audit)


def review_map(*record_ids):
    return {
        record_id: HumanReviewDecision(
            reviewer="human-reviewer",
            human_reviewed=True,
            review_notes="Reviewed for this integration test.",
        )
        for record_id in record_ids
    }


def privacy_review_map(*record_ids):
    return {
        record_id: PrivacyReviewDecision(
            reviewer="privacy-reviewer",
            human_reviewed=True,
            clear=True,
            review_notes="Synthetic independent privacy review.",
        )
        for record_id in record_ids
    }


def metadata_map(*record_ids):
    return {
        record_id: RecordGovernanceMetadata(
            authored_at="2026-11-25T08:00:00Z",
            source_type="dataset",
            group_id=record_id,
        )
        for record_id in record_ids
    }


def govern(records, **kwargs: Any):
    ids = [record.record_id for record in records]
    split_options: dict[str, Any] = {
        "train_ratio": 1.0,
        "validation_ratio": 0.0,
        "test_ratio": 0.0,
    }
    split_options.update(kwargs)
    return govern_prepared_dataset(
        prepared_dataset(*records),
        approved_registry(),
        reviews=review_map(*ids),
        metadata=metadata_map(*ids),
        privacy_reviews=privacy_review_map(*ids),
        **split_options,
    )


def test_explicitly_reviewed_record_enters_governance_manifest():
    result = govern([pipeline_record()])

    assert result.manifest["record_count"] == 1
    assert result.manifest["training_authorized"] is False
    assert result.records[0].approval_status == "approved"
    assert result.records[0].review_status == "reviewed"
    assert len(result.training_records) == 1


def test_legacy_approval_flag_does_not_replace_human_review():
    record = pipeline_record()
    with pytest.raises(GovernanceError, match="review decision missing"):
        govern_prepared_dataset(
            prepared_dataset(record),
            approved_registry(),
            reviews={},
            metadata=metadata_map(record.record_id),
        )


def test_missing_independent_privacy_review_blocks_governance():
    record = pipeline_record()
    with pytest.raises(
        GovernanceError, match="Independent privacy review decision missing"
    ):
        govern_prepared_dataset(
            prepared_dataset(record),
            approved_registry(),
            reviews=review_map(record.record_id),
            metadata=metadata_map(record.record_id),
            privacy_reviews={},
        )


def test_source_must_be_registered_and_approved():
    record = pipeline_record()
    registry = SourceRegistry([
        SourceRegistration(
            source_id="approved-source",
            source_name="Approved Example Dataset",
            source_uri=SOURCE_URI,
            revision="revision-1",
            license_id="CC-BY-4.0",
            license_status="pending",
            approval_status="pending",
            training_allowed=False,
        )
    ])

    with pytest.raises(GovernanceError, match="not explicitly approved"):
        govern_prepared_dataset(
            prepared_dataset(record),
            registry,
            reviews=review_map(record.record_id),
            metadata=metadata_map(record.record_id),
        )


def test_source_uri_must_match_registry():
    record = pipeline_record(source_uri="https://example.org/other")
    with pytest.raises(GovernanceError, match="Source URI mismatch"):
        govern([record])


def test_license_must_match_registry():
    record = pipeline_record(license="MIT")
    with pytest.raises(GovernanceError, match="Licence mismatch"):
        govern([record])


def test_revision_must_match_registry_when_provided():
    record = pipeline_record(dataset_revision="revision-2")
    with pytest.raises(GovernanceError, match="Source revision mismatch"):
        govern([record])


def test_old_pipeline_approval_flag_is_still_required_as_eligibility():
    record = pipeline_record(approved_for_training=False)
    with pytest.raises(GovernanceError, match="not eligible"):
        govern([record])


def test_failed_legacy_audit_blocks_governance():
    record = pipeline_record()
    audit = replace(prepared_dataset(record).audit, privacy_scan_passed=False)
    prepared = PreparedDataset(records=(record,), audit=audit)

    with pytest.raises(GovernanceError, match="audit did not pass"):
        govern_prepared_dataset(
            prepared,
            approved_registry(),
            reviews=review_map(record.record_id),
            metadata=metadata_map(record.record_id),
        )


def test_missing_language_metadata_is_rejected():
    record = pipeline_record(language="")
    with pytest.raises(GovernanceError, match="Language and script"):
        govern([record])


def test_missing_authored_at_metadata_is_rejected():
    record = pipeline_record()
    details = RecordGovernanceMetadata(authored_at="")
    with pytest.raises(GovernanceError, match="Authored-at timestamp"):
        govern_prepared_dataset(
            prepared_dataset(record),
            approved_registry(),
            reviews=review_map(record.record_id),
            metadata={record.record_id: details},
        )


def test_existing_pipeline_split_is_not_reused():
    record = pipeline_record(split="test")
    result = govern([record])

    assert result.manifest["records"][0]["split"] == "train"
    assert len(result.training_records) == 1


def test_test_split_records_are_never_returned_for_training():
    record = pipeline_record()
    result = govern(
        [record],
        train_ratio=0.0,
        validation_ratio=0.0,
        test_ratio=1.0,
    )

    assert result.manifest["records"][0]["split"] == "test"
    assert result.training_records == ()


def test_duplicate_record_ids_are_rejected():
    first = pipeline_record("duplicate-id")
    second = pipeline_record("duplicate-id", text="A different sufficiently long example.")
    with pytest.raises(GovernanceError, match="Duplicate pipeline record_id"):
        govern([first, second])


def test_related_records_share_a_split():
    first = pipeline_record("related-1")
    second = pipeline_record("related-2", text="Another sufficiently long related example.")
    reviews = review_map("related-1", "related-2")
    metadata = {
        "related-1": RecordGovernanceMetadata(
            authored_at="2026-11-25T08:00:00Z", group_id="shared-group"
        ),
        "related-2": RecordGovernanceMetadata(
            authored_at="2026-11-25T08:00:00Z", group_id="shared-group"
        ),
    }

    result = govern_prepared_dataset(
        prepared_dataset(first, second),
        approved_registry(),
        reviews=reviews,
        metadata=metadata,
        privacy_reviews=privacy_review_map(*reviews),
        train_ratio=0.5,
        validation_ratio=0.5,
        test_ratio=0.0,
    )

    splits = {entry["split"] for entry in result.manifest["records"]}
    assert len(splits) == 1


def test_duplicate_content_is_blocked_by_manifest():
    first = pipeline_record("duplicate-1")
    second = pipeline_record("duplicate-2")
    reviews = review_map("duplicate-1", "duplicate-2")
    metadata = metadata_map("duplicate-1", "duplicate-2")

    with pytest.raises(GovernanceError, match="Duplicate content"):
        govern_prepared_dataset(
            prepared_dataset(first, second),
            approved_registry(),
            reviews=reviews,
            metadata=metadata,
            privacy_reviews=privacy_review_map(*reviews),
            train_ratio=1.0,
            validation_ratio=0.0,
            test_ratio=0.0,
        )


def synthetic_rights_evidence() -> dict[str, dict[str, str]]:
    """Test-only evidence; it does not represent real source approval."""
    gates = (
        "database_license_scope",
        "underlying_content_rights",
        "collection_terms_ai_ml",
        "attribution_obligations",
        "privacy_personal_data",
        "removal_opt_out",
        "provenance_reproducibility",
        "jurisdiction_legal_signoff",
    )
    return {
        gate: {
            "status": "verified",
            "evidence_reference": f"test://synthetic-rights/{gate}",
            "reviewer": "synthetic-test-reviewer",
            "reviewed_at": "2026-11-27T00:00:00+00:00",
        }
        for gate in gates
    }

