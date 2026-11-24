import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from aviy.data import DatasetRecord, prepare_dataset, write_jsonl
from aviy.data.audit import audit_dataset
from aviy.tokenizer import ByteBPETokenizer
from aviy.tokenizer.benchmark import benchmark_tokenizer


class TokenizerHardeningTests(unittest.TestCase):
    def test_training_consumes_generator_once(self):
        consumed = []
        def samples():
            for value in ("alpha beta", "alpha gamma"):
                consumed.append(value)
                yield value

        tokenizer = ByteBPETokenizer.train(
            samples(), vocabulary_size=270, minimum_pair_frequency=1
        )
        self.assertGreaterEqual(tokenizer.vocabulary_size, 256)
        self.assertEqual(consumed, ["alpha beta", "alpha gamma"])

    def test_training_rejects_invalid_parameters(self):
        for value in (True, 255, 256.5, "300"):
            with self.subTest(vocabulary_size=value), self.assertRaises(ValueError):
                ByteBPETokenizer.train(["sample"], vocabulary_size=value)
        for value in (True, 0, 1.5, "2"):
            with self.subTest(minimum_pair_frequency=value), self.assertRaises(ValueError):
                ByteBPETokenizer.train(
                    ["sample"], minimum_pair_frequency=value
                )

    def test_training_rejects_non_string_samples(self):
        with self.assertRaises(TypeError):
            ByteBPETokenizer.train(["valid", 123])

    def test_decode_rejects_non_integer_token_ids(self):
        tokenizer = ByteBPETokenizer()
        for token in (True, 1.0, "65", None):
            with self.subTest(token=token), self.assertRaises(TypeError):
                tokenizer.decode([token])

    def test_decode_rejects_unknown_and_invalid_utf8_ids(self):
        tokenizer = ByteBPETokenizer()
        with self.assertRaises(ValueError):
            tokenizer.decode([999])
        with self.assertRaises(ValueError):
            tokenizer.decode([255])

    def test_decode_rejects_non_iterable(self):
        with self.assertRaises(TypeError):
            ByteBPETokenizer().decode(12)

    def test_training_merges_are_deterministic(self):
        texts = [
            "alpha beta gamma",
            "alpha beta delta",
            "repeatable tokenizer training",
            "repeatable tokenizer training",
        ]
        first = ByteBPETokenizer.train(
            texts, vocabulary_size=280, minimum_pair_frequency=1
        )
        second = ByteBPETokenizer.train(
            texts, vocabulary_size=280, minimum_pair_frequency=1
        )
        self.assertEqual(first.merges, second.merges)
        self.assertEqual(first.vocabulary_size, second.vocabulary_size)

    def test_benchmark_hashes_are_deterministic_and_sensitive(self):
        tokenizer = ByteBPETokenizer.train(
            ["alpha beta", "alpha gamma"],
            vocabulary_size=270,
            minimum_pair_frequency=1,
        )
        first = benchmark_tokenizer(tokenizer, ["alpha beta", "hello"])
        second = benchmark_tokenizer(tokenizer, ["alpha beta", "hello"])
        changed = benchmark_tokenizer(tokenizer, ["alpha beta", "changed"])
        self.assertEqual(first["tokenizer_sha256"], second["tokenizer_sha256"])
        self.assertEqual(first["corpus_sha256"], second["corpus_sha256"])
        self.assertNotEqual(first["corpus_sha256"], changed["corpus_sha256"])
        self.assertEqual(first["sample_count"], 2)


class DatasetHardeningTests(unittest.TestCase):
    def record(self, text="A valid dataset example.", **kwargs):
        values = {
            "text": text,
            "source_id": "fixture-v1",
            "license": "CC-BY-4.0",
            "approved_for_training": True,
            "source_uri": "dataset:fixture-v1",
        }
        values.update(kwargs)
        return DatasetRecord(**values)

    def test_rejects_non_finite_or_boolean_split_ratios(self):
        for value in (True, float("nan"), float("inf"), -0.1):
            with self.subTest(ratio=value), self.assertRaises(ValueError):
                prepare_dataset(
                    [self.record()],
                    allowed_licenses=["CC-BY-4.0"],
                    train_ratio=value,
                    validation_ratio=0.0,
                    test_ratio=1.0 - value if value is not True and value == 0 else 0.0,
                )

    def test_rejects_invalid_character_limits(self):
        for limits in (
            {"minimum_characters": True},
            {"maximum_characters": 100.5},
            {"minimum_characters": 0},
            {"minimum_characters": 20, "maximum_characters": 10},
        ):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                prepare_dataset(
                    [self.record()],
                    allowed_licenses=["CC-BY-4.0"],
                    **limits,
                )

    def test_rejects_non_string_allowed_license(self):
        with self.assertRaises(TypeError):
            prepare_dataset(
                [self.record()],
                allowed_licenses=["CC-BY-4.0", 123],
            )

    def test_rejects_non_string_record_id(self):
        result = prepare_dataset(
            [self.record(record_id=123)],
            allowed_licenses=["CC-BY-4.0"],
        )
        self.assertEqual(result.audit.accepted, 0)
        self.assertFalse(result.audit.provenance_complete)

    def test_audit_records_exact_source_and_prepared_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source.jsonl"
            output = base / "audit"
            write_jsonl(
                [self.record(f"Dataset example number {index}.") for index in range(20)],
                source,
            )
            report = audit_dataset(
                source,
                output,
                allowed_licenses=["CC-BY-4.0"],
                seed=42,
            )
            self.assertEqual(
                report["source_sha256"],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                report["prepared_sha256"],
                hashlib.sha256((output / "prepared.jsonl").read_bytes()).hexdigest(),
            )
            self.assertEqual(report["audit_schema"], "aviy-dataset-audit-v2")
            saved = json.loads((output / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, report)

    def test_audit_hash_changes_when_source_bytes_change(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source.jsonl"
            write_jsonl([self.record()], source)
            first = audit_dataset(
                source, base / "audit1", allowed_licenses=["CC-BY-4.0"]
            )
            source.write_bytes(source.read_bytes() + b"\n")
            second = audit_dataset(
                source, base / "audit2", allowed_licenses=["CC-BY-4.0"]
            )
            self.assertNotEqual(first["source_sha256"], second["source_sha256"])


if __name__ == "__main__":
    unittest.main()
