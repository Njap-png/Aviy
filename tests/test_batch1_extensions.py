from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from aviy.models.base_model import AviyBaseLM, BaseModelConfig
from aviy.tokenizer import (
    ByteBPETokenizer,
    benchmark_tokenizer,
    write_tokenizer_benchmark,
)
from aviy.training import (
    ExperimentTracker,
    load_verified_checkpoint,
    train_steps,
    verify_checkpoint,
)


class BatchOneExtensionTests(unittest.TestCase):
    def _model(self) -> AviyBaseLM:
        return AviyBaseLM(
            BaseModelConfig(
                vocab_size=64,
                context_length=8,
                d_model=16,
                n_layers=1,
                n_heads=2,
                d_ff=32,
                dropout=0.0,
            )
        )

    def _batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(123)
        inputs = torch.randint(0, 64, (2, 6), generator=generator)
        targets = torch.randint(0, 64, (2, 6), generator=generator)
        return inputs, targets

    def test_tokenizer_coverage_and_benchmark(self) -> None:
        texts = [
            "Hello world",
            "Caf\u00e9 d\u00e9j\u00e0 vu",
            "\u0393\u03b5\u03b9\u03ac \u03c3\u03bf\u03c5",
            "Kiswahili ni lugha nzuri",
            "\u0645\u0631\u062d\u0628\u0627",
            "\u4f60\u597d\u4e16\u754c",
        ]
        tokenizer = ByteBPETokenizer.train(
            texts,
            vocabulary_size=300,
            minimum_pair_frequency=1,
        )
        metrics = tokenizer.measure(texts)
        self.assertEqual(metrics["character_coverage_rate"], 1.0)
        self.assertEqual(metrics["byte_coverage_rate"], 1.0)
        self.assertEqual(metrics["unknown_characters"], 0)
        self.assertEqual(metrics["round_trip_rate"], 1.0)

        report = benchmark_tokenizer(tokenizer, texts)
        self.assertEqual(report["metrics"]["round_trip_rate"], 1.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer_benchmark.json"
            write_tokenizer_benchmark(tokenizer, texts, path)
            self.assertTrue(path.is_file())

    def test_checkpoint_has_integrity_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model = self._model()
            batch = self._batch()

            train_steps(
                model,
                [batch],
                steps=1,
                seed=42,
                device_name="cpu",
                precision="fp32",
                checkpoint_path=path,
            )

            self.assertTrue(path.is_file())
            self.assertTrue(
                Path(str(path) + ".sha256").is_file()
            )
            self.assertEqual(len(verify_checkpoint(path)), 64)
            payload = load_verified_checkpoint(path)
            self.assertEqual(payload["step"], 1)

    def test_corrupted_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model = self._model()
            train_steps(
                model,
                [self._batch()],
                steps=1,
                seed=42,
                device_name="cpu",
                precision="fp32",
                checkpoint_path=path,
            )

            with path.open("ab") as stream:
                stream.write(b"corruption")

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_checkpoint(path)

    def test_interrupted_training_resume_matches_continuous_run(self) -> None:
        batch = self._batch()

        torch.manual_seed(77)
        continuous = self._model()
        train_steps(
            continuous,
            [batch],
            steps=4,
            seed=77,
            device_name="cpu",
            precision="fp32",
        )

        torch.manual_seed(77)
        interrupted = self._model()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resume.pt"

            first = train_steps(
                interrupted,
                [batch],
                steps=2,
                seed=77,
                device_name="cpu",
                precision="fp32",
                checkpoint_path=checkpoint,
            )
            self.assertEqual(first["global_step"], 2)

            resumed = train_steps(
                interrupted,
                [batch],
                steps=2,
                device_name="cpu",
                precision="fp32",
                resume_from_checkpoint=checkpoint,
            )
            self.assertEqual(resumed["resumed_from_step"], 2)
            self.assertEqual(resumed["global_step"], 4)

        for expected, actual in zip(
            continuous.state_dict().values(),
            interrupted.state_dict().values(),
            strict=True,
        ):
            self.assertTrue(torch.equal(expected, actual))


    def test_multi_batch_resume_preserves_batch_position(self) -> None:
        batches = [
            self._batch(),
            (
                torch.full((2, 6), 3, dtype=torch.long),
                torch.full((2, 6), 4, dtype=torch.long),
            ),
            (
                torch.full((2, 6), 5, dtype=torch.long),
                torch.full((2, 6), 6, dtype=torch.long),
            ),
        ]

        torch.manual_seed(91)
        continuous = self._model()
        train_steps(
            continuous,
            batches,
            steps=7,
            seed=91,
            device_name="cpu",
            precision="fp32",
        )

        torch.manual_seed(91)
        interrupted = self._model()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "multi_batch_resume.pt"

            first = train_steps(
                interrupted,
                batches,
                steps=2,
                seed=91,
                device_name="cpu",
                precision="fp32",
                checkpoint_path=checkpoint,
            )
            self.assertEqual(first["next_batch_position"], 2)

            resumed = train_steps(
                interrupted,
                batches,
                steps=5,
                device_name="cpu",
                precision="fp32",
                resume_from_checkpoint=checkpoint,
            )
            self.assertEqual(resumed["resumed_from_step"], 2)
            self.assertEqual(resumed["global_step"], 7)
            self.assertEqual(resumed["next_batch_position"], 1)

        for expected, actual in zip(
            continuous.state_dict().values(),
            interrupted.state_dict().values(),
            strict=True,
        ):
            self.assertTrue(torch.equal(expected, actual))

    def test_missing_checkpoint_sidecar_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            train_steps(
                self._model(),
                [self._batch()],
                steps=1,
                seed=42,
                device_name="cpu",
                precision="fp32",
                checkpoint_path=path,
            )

            Path(str(path) + ".sha256").unlink()

            with self.assertRaisesRegex(ValueError, "sidecar is missing"):
                verify_checkpoint(path)


    def test_experiment_metadata_and_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = ExperimentTracker(
                root / "parent",
                config={"learning_rate": 1e-4},
                seed=42,
                description="Baseline",
            )
            child = ExperimentTracker(
                root / "child",
                config={"learning_rate": 5e-5},
                seed=42,
                parent_run_id=parent.run_id,
                description="Candidate",
            )
            child.record_result({"accuracy": 0.75})

            import json

            metadata = json.loads(
                (root / "child" / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["parent_run_id"], parent.run_id)
            self.assertEqual(metadata["run_id"], child.run_id)
            self.assertTrue(
                (root / "child" / "events.jsonl").is_file()
            )


if __name__ == "__main__":
    unittest.main()
