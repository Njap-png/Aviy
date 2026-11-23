import tempfile
import unittest
from pathlib import Path

import torch

from aviy.data import DatasetRecord, audit_dataset, write_jsonl
from aviy.models.base_model import (
    AviyBaseLM,
    BaseModelConfig,
    validate_tokenizer_compatibility,
)
from aviy.tokenizer import ByteBPETokenizer
from aviy.training import train_steps


class TokenizerTests(unittest.TestCase):
    def test_multilingual_round_trip(self):
        texts = [
            "Hello world",
            "Caf\u00e9 d\u00e9j\u00e0 vu",
            "\u0393\u03b5\u03b9\u03ac \u03c3\u03bf\u03c5",
            "Kiswahili ni lugha nzuri",
            "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645",
            "\u4f60\u597d\u4e16\u754c",
        ]
        tokenizer = ByteBPETokenizer.train(
            texts, vocabulary_size=300, minimum_pair_frequency=1
        )
        for sample in texts:
            normalized = tokenizer.normalize(sample)
            self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), normalized)
        self.assertEqual(tokenizer.measure(texts)["round_trip_rate"], 1.0)

    def test_save_and_load(self):
        tokenizer = ByteBPETokenizer.train(
            ["alpha beta", "alpha gamma"], vocabulary_size=270,
            minimum_pair_frequency=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            loaded = ByteBPETokenizer.load(path)
            self.assertEqual(loaded.encode("alpha beta"), tokenizer.encode("alpha beta"))

    def test_requires_training_text(self):
        with self.assertRaises(ValueError):
            ByteBPETokenizer.train([], vocabulary_size=300)

    def test_trains_from_audited_training_split(self):
        records = [
            DatasetRecord(
                text=f"Approved multilingual sample {index}",
                source_id="fixture",
                license="CC-BY-4.0",
                approved_for_training=True,
                source_uri="test://fixture",
            )
            for index in range(40)
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            write_jsonl(records, source)
            prepared = audit_dataset(
                source, Path(directory) / "audit",
                allowed_licenses=["CC-BY-4.0"], seed=42,
            )
            self.assertTrue(prepared["passed"])
            from aviy.data.pipeline import prepare_dataset
            audited = prepare_dataset(
                records, allowed_licenses=["CC-BY-4.0"], seed=42
            )
            tokenizer = ByteBPETokenizer.train_from_prepared_dataset(
                audited, vocabulary_size=300, minimum_pair_frequency=1
            )
            self.assertGreaterEqual(tokenizer.vocabulary_size, 256)



class DatasetAuditTests(unittest.TestCase):
    def test_audit_writes_reproducible_artifacts(self):
        records = [
            DatasetRecord(
                text=f"Approved example record {index}",
                source_id="synthetic-test-fixture",
                license="CC-BY-4.0",
                approved_for_training=True,
                source_uri="test://fixture",
            )
            for index in range(20)
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            output = Path(directory) / "audit"
            write_jsonl(records, source)
            report = audit_dataset(
                source, output, allowed_licenses=["CC-BY-4.0"], seed=42
            )
            self.assertTrue(report["passed"])
            self.assertTrue((output / "audit.json").exists())
            self.assertTrue((output / "prepared.jsonl").exists())


class BaseModelTests(unittest.TestCase):
    def test_random_initialization_and_forward(self):
        config = BaseModelConfig(
            vocab_size=300, context_length=32, d_model=32,
            n_layers=2, n_heads=4, d_ff=64,
        )
        model = AviyBaseLM(config)
        output = model(torch.randint(0, 300, (2, 12)))
        self.assertEqual(tuple(output.shape), (2, 12, 300))
        self.assertTrue(model.initialization_verified)
        self.assertGreater(model.parameter_count(), 0)
        self.assertFalse(model.configuration_report()["pretrained_weights_loaded"])
        validate_tokenizer_compatibility(
            ByteBPETokenizer(), config
        )

    def test_rejects_incompatible_tokenizer_vocabulary(self):
        config = BaseModelConfig(
            vocab_size=256, context_length=8, d_model=16,
            n_layers=1, n_heads=2, d_ff=32,
        )
        tokenizer = ByteBPETokenizer([(97, 98)])
        with self.assertRaises(ValueError):
            validate_tokenizer_compatibility(tokenizer, config)

    def test_rejects_pretrained_configuration(self):
        with self.assertRaises(ValueError):
            AviyBaseLM(BaseModelConfig(pretrained_weights=True))


class TrainingEngineTests(unittest.TestCase):
    def test_cpu_smoke_training_and_checkpoint(self):
        torch.manual_seed(42)
        model = AviyBaseLM(BaseModelConfig(
            vocab_size=32, context_length=8, d_model=16,
            n_layers=1, n_heads=2, d_ff=32,
        ))
        inputs = torch.randint(0, 32, (2, 8))
        targets = torch.randint(0, 32, (2, 8))
        batches = [(inputs, targets)]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "smoke.pt"
            report = train_steps(
                model, batches, steps=2, device_name="cpu",
                checkpoint_path=checkpoint,
            )
            self.assertEqual(report["steps"], 2)
            self.assertTrue(checkpoint.exists())
            self.assertTrue(all(torch.isfinite(torch.tensor(report["losses"]))))


    def test_cpu_training_is_reproducible(self):
        config = BaseModelConfig(
            vocab_size=32,
            context_length=8,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            dropout=0.0,
        )

        torch.manual_seed(1234)
        first_model = AviyBaseLM(config)

        torch.manual_seed(1234)
        second_model = AviyBaseLM(config)

        generator = torch.Generator().manual_seed(5678)
        inputs = torch.randint(0, 32, (2, 8), generator=generator)
        targets = torch.randint(0, 32, (2, 8), generator=generator)
        batches = [(inputs, targets)]

        first_report = train_steps(
            first_model,
            batches,
            steps=4,
            learning_rate=1e-4,
            device_name="cpu",
            seed=42,
            precision="fp32",
        )
        second_report = train_steps(
            second_model,
            batches,
            steps=4,
            learning_rate=1e-4,
            device_name="cpu",
            seed=42,
            precision="fp32",
        )

        self.assertEqual(first_report["losses"], second_report["losses"])

        first_state = first_model.state_dict()
        second_state = second_model.state_dict()
        self.assertEqual(first_state.keys(), second_state.keys())

        for name in first_state:
            self.assertTrue(
                torch.equal(first_state[name], second_state[name]),
                f"Model parameter differs between runs: {name}",
            )


if __name__ == "__main__":
    unittest.main()
