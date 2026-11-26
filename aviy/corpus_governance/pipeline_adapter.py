"""Strict adapter between Aviy's dataset pipeline and corpus governance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from aviy.data.pipeline import PreparedDataset
from aviy.corpus_governance.engine import (
    CorpusRecord,
    GovernanceError,
    SourceRegistry,
    approve_record,
    build_manifest,
    create_record,
    training_records,
)


@dataclass(frozen=True)
class HumanReviewDecision:
    """Explicit per-record human review evidence."""

    reviewer: str
    human_reviewed: bool
    review_notes: str = ""


@dataclass(frozen=True)
class RecordGovernanceMetadata:
    """Required metadata not represented by the legacy pipeline record."""

    authored_at: str
    source_type: str = "dataset"
    group_id: str = ""


@dataclass(frozen=True)
class GovernedDataset:
    """Approved records and their authoritative governance manifest."""

    records: tuple[CorpusRecord, ...]
    manifest: dict
    training_records: tuple[CorpusRecord, ...]


def govern_prepared_dataset(
    prepared: PreparedDataset,
    registry: SourceRegistry,
    *,
    reviews: Mapping[str, HumanReviewDecision],
    metadata: Mapping[str, RecordGovernanceMetadata],
    seed: int = 42,
    train_ratio: float = 0.90,
    validation_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> GovernedDataset:
    """Apply source and human-review gates to an existing prepared dataset.

    The legacy pipeline's approved_for_training flag is treated only as an
    eligibility signal. It never substitutes for source approval or explicit
    human review. The governance manifest independently assigns all splits.
    """
    if not isinstance(prepared, PreparedDataset):
        raise TypeError("prepared must be PreparedDataset")
    if not isinstance(registry, SourceRegistry):
        raise TypeError("registry must be SourceRegistry")
    if not prepared.audit.passed:
        raise GovernanceError("Legacy dataset audit did not pass")
    if not prepared.records:
        raise GovernanceError("Prepared dataset contains no records")

    if not isinstance(reviews, Mapping):
        raise TypeError("reviews must be a mapping")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")

    seen_ids: set[str] = set()
    approved_records: list[CorpusRecord] = []

    for item in prepared.records:
        record_id = item.record_id.strip()
        if not record_id:
            raise GovernanceError("Every pipeline record needs a stable record_id")
        if record_id in seen_ids:
            raise GovernanceError("Duplicate pipeline record_id")
        seen_ids.add(record_id)

        if item.approved_for_training is not True:
            raise GovernanceError(
                f"Pipeline record {record_id} is not eligible for governance review"
            )

        source = registry.get(item.source_id)
        if source is None:
            raise GovernanceError(f"Unregistered source: {item.source_id}")
        if not registry.is_training_approved(item.source_id):
            raise GovernanceError(
                f"Source is not explicitly approved for training: {item.source_id}"
            )

        if item.license.strip().casefold() != source.license_id.strip().casefold():
            raise GovernanceError(f"Licence mismatch for record {record_id}")
        if item.source_uri.strip() != source.source_uri.strip():
            raise GovernanceError(f"Source URI mismatch for record {record_id}")
        if (
            item.dataset_revision.strip()
            and item.dataset_revision.strip() != source.revision.strip()
        ):
            raise GovernanceError(f"Source revision mismatch for record {record_id}")

        review = reviews.get(record_id)
        if not isinstance(review, HumanReviewDecision):
            raise GovernanceError(
                f"Explicit human review decision missing for record {record_id}"
            )

        details = metadata.get(record_id)
        if not isinstance(details, RecordGovernanceMetadata):
            raise GovernanceError(
                f"Required governance metadata missing for record {record_id}"
            )

        if not item.language.strip() or not item.language_script.strip():
            raise GovernanceError(
                f"Language and script are required for record {record_id}"
            )
        if not item.content_category.strip():
            raise GovernanceError(
                f"Content category is required for record {record_id}"
            )
        if not details.authored_at.strip():
            raise GovernanceError(
                f"Authored-at timestamp is required for record {record_id}"
            )
        if not details.source_type.strip():
            raise GovernanceError(
                f"Source type is required for record {record_id}"
            )

        corpus_record = create_record(
            record_id=record_id,
            text=item.text,
            language=item.language,
            script=item.language_script,
            category=item.content_category,
            source_type=details.source_type,
            source_id=item.source_id,
            source_name=source.source_name,
            source_uri=item.source_uri,
            source_revision=source.revision,
            license_id=item.license,
            attribution=source.attribution_text,
            group_id=details.group_id or record_id,
            authored_at=details.authored_at,
        )

        approved_records.append(
            approve_record(
                corpus_record,
                registry,
                reviewer=review.reviewer,
                human_reviewed=review.human_reviewed,
                review_notes=review.review_notes,
            )
        )

    manifest = build_manifest(
        approved_records,
        registry,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
    selected_for_training = training_records(manifest, approved_records)

    return GovernedDataset(
        records=tuple(approved_records),
        manifest=manifest,
        training_records=selected_for_training,
    )
