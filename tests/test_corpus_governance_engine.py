import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aviy.corpus_governance import (
    CorpusRecord,
    GovernanceError,
    SourceRegistration,
    SourceRegistry,
    approve_record,
    build_manifest,
    create_record,
    privacy_issues,
    quarantine_record,
    training_records,
    validate_record,
)
from aviy.corpus_governance.engine import approve_privacy_review


class CorpusGovernanceEngineTests(unittest.TestCase):
    def setUp(self):
        self.registry = SourceRegistry([
            SourceRegistration(
                source_id="aviy-original-v1",
                source_name="Aviy Original",
                source_uri="aviy://original",
                revision="v1",
                license_id="Aviy-Original",
                license_status="approved",
                approval_status="approved",
                training_allowed=True,
                attribution_required=False,
                reviewer="governance-reviewer",
                approved_at="2026-11-26T00:00:00+00:00",
                rights_evidence=synthetic_rights_evidence(),
            )
        ])

    def record(self, number=1, **overrides):
        values = {
            "record_id": f"record-{number}",
            "text": f"This is an original educational explanation number {number}.",
            "language": "en",
            "script": "Latn",
            "category": "general_english",
            "source_type": "original",
            "source_id": "aviy-original-v1",
            "source_name": "Aviy Original",
            "source_uri": "aviy://original/lesson",
            "source_revision": "v1",
            "license_id": "Aviy-Original",
            "attribution": "",
            "group_id": f"document-{number}",
        }
        values.update(overrides)
        return create_record(**values)

    def approved(self, record):
        content_approved = approve_record(
            record, self.registry,
            reviewer="human-reviewer",
            human_reviewed=True,
        )
        return approve_privacy_review(
            content_approved,
            reviewer="privacy-reviewer",
            human_reviewed=True,
            clear=True,
            review_notes="Synthetic privacy review fixture.",
        )

    def test_valid_record_requires_human_approval(self):
        record = self.record()
        self.assertEqual(validate_record(record, self.registry), ())
        approved = self.approved(record)
        self.assertEqual(approved.approval_status, "approved")
        self.assertEqual(approved.review_status, "reviewed")
        self.assertEqual(approved.privacy_status, "clear")


    def test_content_approval_alone_does_not_clear_privacy(self):
        content_approved = approve_record(
            self.record(),
            self.registry,
            reviewer="content-reviewer",
            human_reviewed=True,
        )
        self.assertEqual(content_approved.approval_status, "approved")
        self.assertEqual(content_approved.review_status, "reviewed")
        self.assertEqual(content_approved.privacy_status, "pending")
        self.assertEqual(content_approved.privacy_reviewer, "")
        with self.assertRaisesRegex(
            GovernanceError, "independent privacy-review evidence missing"
        ):
            build_manifest([content_approved], self.registry)

    def test_manifest_rejects_same_content_and_privacy_reviewer(self):
        content_approved = approve_record(
            self.record(),
            self.registry,
            reviewer="content-reviewer",
            human_reviewed=True,
        )
        forged_review = replace(
            content_approved,
            privacy_status="clear",
            privacy_reviewer=" CONTENT-REVIEWER ",
            privacy_reviewed_at="2026-11-27T00:00:00+00:00",
        )
        with self.assertRaisesRegex(
            GovernanceError, "privacy reviewer must be independent"
        ):
            build_manifest([forged_review], self.registry)

    def test_privacy_review_requires_explicit_human_decision(self):
        content_approved = approve_record(
            self.record(),
            self.registry,
            reviewer="content-reviewer",
            human_reviewed=True,
        )
        with self.assertRaisesRegex(GovernanceError, "human privacy review"):
            approve_privacy_review(
                content_approved,
                reviewer="privacy-reviewer",
                human_reviewed=False,
                clear=True,
            )

    def test_privacy_review_requires_a_different_reviewer(self):
        content_approved = approve_record(
            self.record(),
            self.registry,
            reviewer="Content-Reviewer",
            human_reviewed=True,
        )
        with self.assertRaisesRegex(
            GovernanceError, "independent from the content reviewer"
        ):
            approve_privacy_review(
                content_approved,
                reviewer=" content-reviewer ",
                human_reviewed=True,
                clear=True,
                review_notes="Must be independently reviewed.",
            )

    def test_privacy_flag_decision_does_not_enter_manifest(self):
        content_approved = approve_record(
            self.record(),
            self.registry,
            reviewer="content-reviewer",
            human_reviewed=True,
        )
        flagged = approve_privacy_review(
            content_approved,
            reviewer="privacy-reviewer",
            human_reviewed=True,
            clear=False,
            review_notes="Synthetic test flag.",
        )
        self.assertEqual(flagged.privacy_status, "flagged")
        with self.assertRaisesRegex(
            GovernanceError, "independent privacy-review evidence missing"
        ):
            build_manifest([flagged], self.registry)

    def test_automatic_approval_is_forbidden(self):
        with self.assertRaises(GovernanceError):
            approve_record(
                self.record(), self.registry,
                reviewer="reviewer", human_reviewed=False,
            )

    def test_unregistered_source_is_blocked(self):
        issues = validate_record(
            self.record(source_id="unknown"), self.registry
        )
        self.assertIn("unregistered_source", issues)

    def test_unapproved_source_is_blocked(self):
        registry = SourceRegistry([
            SourceRegistration(
                source_id="pending", source_name="Pending",
                source_uri="aviy://pending", revision="v1",
                license_id="L", license_status="pending",
                approval_status="pending", training_allowed=False,
            )
        ])
        record = self.record(
            source_id="pending", source_name="Pending",
            source_uri="aviy://pending/item", license_id="L",
        )
        with self.assertRaises(GovernanceError):
            approve_record(
                record, registry, reviewer="reviewer", human_reviewed=True
            )

    def test_license_mismatch_is_blocked(self):
        self.assertIn(
            "record_source_license_mismatch",
            validate_record(self.record(license_id="Unknown"), self.registry),
        )

    def test_privacy_patterns_are_detected(self):
        self.assertIn("email_address", privacy_issues("Contact person@example.org"))
        self.assertIn("possible_secret", privacy_issues("api_key='abcdefgh123456'"))
        self.assertIn("private_key", privacy_issues("-----BEGIN PRIVATE KEY-----"))

    def test_privacy_flagged_record_cannot_be_approved(self):
        record = self.record(text="Contact person@example.org for this lesson.")
        with self.assertRaises(GovernanceError):
            self.approved(record)

    def test_hash_tampering_is_detected(self):
        record = self.record()
        changed = CorpusRecord(**{
            **record.__dict__,
            "text": "Changed content without changing its recorded hash.",
        })
        self.assertIn("content_hash_mismatch", validate_record(changed, self.registry))

    def test_quarantined_record_cannot_be_approved(self):
        record = quarantine_record(self.record(), reason="Needs review")
        with self.assertRaises(GovernanceError):
            self.approved(record)

    def test_manifest_rejects_pending_record(self):
        with self.assertRaises(GovernanceError):
            build_manifest([self.record()], self.registry)

    def test_manifest_rejects_duplicate_content(self):
        first = self.approved(self.record(1))
        second = self.approved(self.record(
            2, text=first.text, group_id="document-2"
        ))
        with self.assertRaises(GovernanceError):
            build_manifest([first, second], self.registry)

    def test_related_records_share_a_split(self):
        first = self.approved(self.record(1, group_id="same-document"))
        second = self.approved(self.record(2, group_id="same-document"))
        manifest = build_manifest([first, second], self.registry)
        splits = {entry["split"] for entry in manifest["records"]}
        self.assertEqual(len(splits), 1)

    def test_manifest_is_deterministic_except_timestamp(self):
        records = [self.approved(self.record(i)) for i in range(1, 40)]
        first = build_manifest(records, self.registry, seed=123)
        second = build_manifest(records, self.registry, seed=123)
        self.assertEqual(first["records"], second["records"])
        self.assertEqual(first["split_counts"], second["split_counts"])

    def test_training_accessor_never_returns_validation_or_test(self):
        records = [self.approved(self.record(i)) for i in range(1, 100)]
        manifest = build_manifest(
            records, self.registry, seed=42,
            train_ratio=0.70, validation_ratio=0.15, test_ratio=0.15,
        )
        selected = training_records(manifest, records)
        train_ids = {
            entry["record_id"] for entry in manifest["records"]
            if entry["split"] == "train"
        }
        self.assertEqual({record.record_id for record in selected}, train_ids)
        self.assertTrue(all(record.approval_status == "approved" for record in selected))

    def test_registry_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            self.registry.save(path)
            loaded = SourceRegistry.load(path)
            self.assertTrue(loaded.is_training_approved("aviy-original-v1"))

    def test_empty_attribution_is_blocked_when_required(self):
        source = SourceRegistration(
            source_id="licensed", source_name="Licensed Source",
            source_uri="https://example.org", revision="v1",
            license_id="CC-BY-4.0", license_status="approved",
            approval_status="approved", training_allowed=True,
            attribution_required=True, reviewer="reviewer",
            approved_at="2026-11-26",
            rights_evidence=synthetic_rights_evidence(),
        )
        registry = SourceRegistry([source])
        record = self.record(
            source_id="licensed", source_name="Licensed Source",
            source_uri="https://example.org/page",
            license_id="CC-BY-4.0", attribution="",
        )
        self.assertIn(
            "missing_required_attribution",
            validate_record(record, registry),
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


if __name__ == "__main__":
    unittest.main()
