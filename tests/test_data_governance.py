import tempfile
import unittest
from pathlib import Path

from aviy.data import (
    DatasetRecord,
    normalize_text,
    prepare_dataset,
    privacy_issues,
    read_jsonl,
    write_jsonl,
)


class DatasetGovernanceTests(unittest.TestCase):
    def record(self, text, **kwargs):
        values = {
            "text": text,
            "source_id": "approved-corpus-v1",
            "license": "CC-BY-4.0",
            "source_uri": "dataset:approved-corpus-v1",
            "approved_for_training": True,
        }
        values.update(kwargs)
        return DatasetRecord(**values)

    def test_unicode_and_whitespace_normalization(self):
        self.assertEqual(
            normalize_text("  Cafe" + chr(769) + " \n test  "),
            "Caf" + chr(233) + " test",
        )

    def test_approved_records_are_ingested(self):
        result = prepare_dataset(
            [self.record("A valid example for the training corpus.")],
            allowed_licenses=["CC-BY-4.0"],
        )
        self.assertEqual(result.audit.accepted, 1)
        self.assertTrue(result.audit.passed)

    def test_unapproved_license_is_rejected_and_audited(self):
        result = prepare_dataset(
            [self.record("A valid example.", license="UNKNOWN")],
            allowed_licenses=["CC-BY-4.0"],
        )
        self.assertEqual(result.audit.accepted, 0)
        self.assertFalse(result.audit.licenses_approved)

    def test_unapproved_record_is_rejected(self):
        result = prepare_dataset(
            [self.record("A valid example.", approved_for_training=False)],
            allowed_licenses=["CC-BY-4.0"],
        )
        self.assertEqual(result.audit.accepted, 0)

    def test_missing_provenance_is_rejected(self):
        result = prepare_dataset(
            [self.record("A valid example.", source_id="")],
            allowed_licenses=["CC-BY-4.0"],
        )
        self.assertEqual(result.audit.accepted, 0)
        self.assertFalse(result.audit.provenance_complete)

    def test_missing_source_uri_is_rejected(self):
        result = prepare_dataset(
            [self.record("A valid example.", source_uri="")],
            allowed_licenses=["CC-BY-4.0"],
        )
        self.assertEqual(result.audit.accepted, 0)
        self.assertFalse(result.audit.provenance_complete)

    def test_malformed_record_types_are_rejected(self):
        result = prepare_dataset(
            [
                {
                    "text": 123,
                    "source_id": "approved-corpus-v1",
                    "license": "CC-BY-4.0",
                    "source_uri": "dataset:approved-corpus-v1",
                    "approved_for_training": True,
                }
            ],
            allowed_licenses=["CC-BY-4.0"],
        )
        self.assertEqual(result.audit.accepted, 0)
        self.assertFalse(result.audit.provenance_complete)

    def test_privacy_patterns_are_detected(self):
        self.assertIn("email_address", privacy_issues("Contact person@example.org"))
        self.assertIn("possible_secret", privacy_issues("api_key = 'abcdefgh123456'"))
        self.assertIn("private_key", privacy_issues("-----BEGIN PRIVATE KEY-----"))

    def test_private_data_is_rejected(self):
        result = prepare_dataset(
            [self.record("Contact person@example.org for details.")],
            allowed_licenses=["CC-BY-4.0"],
        )
        self.assertEqual(result.audit.accepted, 0)
        self.assertFalse(result.audit.privacy_scan_passed)

    def test_duplicate_text_is_removed(self):
        result = prepare_dataset(
            [
                self.record("The same example text."),
                self.record("  THE SAME EXAMPLE TEXT.  "),
            ],
            allowed_licenses=["CC-BY-4.0"],
        )
        self.assertEqual(result.audit.accepted, 1)
        self.assertEqual(result.audit.duplicates_removed, 1)

    def test_splits_are_reproducible(self):
        records = [
            self.record(f"Example training record number {index}.")
            for index in range(100)
        ]
        first = prepare_dataset(records, allowed_licenses=["CC-BY-4.0"], seed=19)
        second = prepare_dataset(records, allowed_licenses=["CC-BY-4.0"], seed=19)
        self.assertEqual(first.records, second.records)
        self.assertTrue(first.audit.split_integrity_passed)

    def test_split_ratios_are_validated(self):
        with self.assertRaises(ValueError):
            prepare_dataset(
                [self.record("A valid example.")],
                allowed_licenses=["CC-BY-4.0"],
                train_ratio=0.8,
                validation_ratio=0.1,
                test_ratio=0.2,
            )

    def test_jsonl_round_trip(self):
        records = [self.record("A JSONL example.")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            write_jsonl(records, path)
            self.assertEqual(read_jsonl(path), records)

    def test_extended_provenance_is_preserved_and_screened(self):
        record = self.record(
            "A valid provenance example for Aviy.",
            source_uri="https://example.org/document/123",
            dataset_revision="0123456789abcdef",
            license_uri="https://opendatacommons.org/licenses/by/1-0/",
            language="sw",
            language_script="Latn",
            retrieved_at="2026-11-25T10:00:00Z",
            training_stage="base_pretraining",
            content_category="general_language",
            processing_history=("source_retrieved",),
        )

        result = prepare_dataset(
            [record],
            allowed_licenses=["CC-BY-4.0"],
        )

        self.assertTrue(result.audit.passed)
        self.assertEqual(result.audit.accepted, 1)

        accepted = result.records[0]
        self.assertEqual(accepted.dataset_revision, "0123456789abcdef")
        self.assertEqual(
            accepted.license_uri,
            "https://opendatacommons.org/licenses/by/1-0/",
        )
        self.assertEqual(accepted.language, "sw")
        self.assertEqual(accepted.language_script, "Latn")
        self.assertEqual(accepted.retrieved_at, "2026-11-25T10:00:00Z")
        self.assertEqual(accepted.training_stage, "base_pretraining")
        self.assertEqual(accepted.content_category, "general_language")
        self.assertEqual(accepted.processing_history, ("source_retrieved",))
        self.assertEqual(accepted.privacy_screening, "automated_checks_passed")
        self.assertEqual(accepted.duplicate_screening, "automated_checks_passed")

    def test_normalization_is_recorded_in_processing_history(self):
        record = self.record(
            "  A normalized example.  ",
            processing_history=("source_retrieved",),
        )

        result = prepare_dataset(
            [record],
            allowed_licenses=["CC-BY-4.0"],
        )

        self.assertEqual(result.audit.accepted, 1)
        self.assertEqual(
            result.records[0].processing_history,
            (
                "source_retrieved",
                "unicode_nfc_whitespace_normalization",
            ),
        )

    def test_extended_metadata_jsonl_round_trip(self):
        record = self.record(
            "A complete metadata round-trip example.",
            dataset_revision="revision-123",
            language="en",
            language_script="Latn",
            training_stage="base_pretraining",
            processing_history=("downloaded", "normalized"),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extended.jsonl"
            write_jsonl([record], path)
            restored = read_jsonl(path)

        self.assertEqual(restored, [record])

    def test_invalid_processing_history_is_rejected(self):
        result = prepare_dataset(
            [{
                "text": "A valid example.",
                "source_id": "approved-corpus-v1",
                "license": "CC-BY-4.0",
                "source_uri": "dataset:approved-corpus-v1",
                "approved_for_training": True,
                "processing_history": ["downloaded", 123],
            }],
            allowed_licenses=["CC-BY-4.0"],
        )

        self.assertEqual(result.audit.accepted, 0)
        self.assertFalse(result.audit.provenance_complete)

    def test_invalid_license_uri_is_rejected(self):
        result = prepare_dataset(
            [
                self.record(
                    "A valid example with an invalid license URI.",
                    license_uri="not a valid URI",
                )
            ],
            allowed_licenses=["CC-BY-4.0"],
        )

        self.assertEqual(result.audit.accepted, 0)
        self.assertFalse(result.audit.provenance_complete)

    def test_malformed_jsonl_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text("{invalid json}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_jsonl(path)


if __name__ == "__main__":
    unittest.main()
