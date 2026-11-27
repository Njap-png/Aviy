from __future__ import annotations

from copy import deepcopy

from aviy.corpus_governance.engine import (
    REQUIRED_RIGHTS_EVIDENCE_GATES,
    SourceRegistration,
    SourceRegistry,
)


def complete_evidence() -> dict[str, dict[str, str]]:
    return {
        gate: {
            "status": "verified",
            "evidence_reference": f"test://rights/{gate}",
            "reviewer": "synthetic-test-reviewer",
            "reviewed_at": "2026-11-27T00:00:00+00:00",
        }
        for gate in REQUIRED_RIGHTS_EVIDENCE_GATES
    }


def make_source(evidence: dict[str, dict[str, str]]) -> SourceRegistration:
    return SourceRegistration(
        source_id="synthetic-source",
        source_name="Synthetic Test Source",
        source_uri="https://example.invalid/dataset",
        revision="synthetic-revision",
        license_id="Synthetic-Test-License",
        license_status="approved",
        approval_status="approved",
        training_allowed=True,
        attribution_required=False,
        reviewer="synthetic-source-reviewer",
        approved_at="2026-11-27T00:00:00+00:00",
        rights_evidence=evidence,
    )


def test_source_requires_all_eight_rights_evidence_gates() -> None:
    assert len(REQUIRED_RIGHTS_EVIDENCE_GATES) == 8
    assert not SourceRegistry([make_source({})]).is_training_approved(
        "synthetic-source"
    )

    complete = complete_evidence()
    assert SourceRegistry([make_source(complete)]).is_training_approved(
        "synthetic-source"
    )

    for gate in REQUIRED_RIGHTS_EVIDENCE_GATES:
        partial = deepcopy(complete)
        del partial[gate]
        assert not SourceRegistry([make_source(partial)]).is_training_approved(
            "synthetic-source"
        ), gate


def test_rights_evidence_requires_verified_status_and_traceable_review() -> None:
    for field, value in (
        ("status", "pending"),
        ("evidence_reference", ""),
        ("reviewer", ""),
        ("reviewed_at", "not-a-timestamp"),
        ("reviewed_at", "2026-11-27T00:00:00"),
    ):
        evidence = complete_evidence()
        evidence["underlying_content_rights"][field] = value
        assert not SourceRegistry([make_source(evidence)]).is_training_approved(
            "synthetic-source"
        ), (field, value)


def test_legacy_registry_remains_loadable_but_is_not_training_approved() -> None:
    legacy = SourceRegistry.from_dict({
        "schema_version": "1.0",
        "sources": [{
            "source_id": "legacy-source",
            "source_name": "Legacy Source",
            "source_uri": "https://example.invalid/legacy",
            "revision": "legacy-revision",
            "license_id": "Legacy-License",
            "license_status": "approved",
            "approval_status": "approved",
            "training_allowed": True,
            "attribution_required": False,
            "reviewer": "legacy-reviewer",
            "approved_at": "2026-11-26T00:00:00+00:00",
        }],
    })
    assert legacy.get("legacy-source") is not None
    assert not legacy.is_training_approved("legacy-source")
