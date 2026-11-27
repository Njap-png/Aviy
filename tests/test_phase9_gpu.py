"""CPU-safe and CUDA-conditional tests for Phase 9 training infrastructure."""

import math
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import torch

from aviy.models.base_model import AviyBaseLM, BaseModelConfig
from aviy.training import train_steps
from aviy.training.engine import resource_report


def make_model() -> AviyBaseLM:
    return AviyBaseLM(
        BaseModelConfig(
            vocab_size=32,
            context_length=8,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            dropout=0.0,
        )
    )


def make_batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(123)
    inputs = torch.randint(0, 32, (2, 8), generator=generator)
    targets = torch.randint(0, 32, (2, 8), generator=generator)
    return [(inputs, targets)]


class Phase9CPUTests(unittest.TestCase):
    def test_cpu_precision_modes_fall_back_safely(self):
        for precision in ("auto", "fp32", "fp16", "bf16"):
            with self.subTest(precision=precision):
                report = train_steps(
                    make_model(),
                    make_batches(),
                    steps=1,
                    device_name="cpu",
                    seed=42,
                    precision=precision,
                )
                resources = cast(dict[str, object], report["resources"])
                losses = cast(list[float], report["losses"])
                self.assertEqual(resources["device"], "cpu")
                self.assertTrue(
                    all(math.isfinite(float(loss)) for loss in losses)
                )
                self.assertNotIn("cuda_allocated_bytes", resources)

    def test_cpu_resource_report(self):
        report = resource_report(torch.device("cpu"))
        resources = cast(dict[str, object], report)
        cpu_threads = cast(int, resources["cpu_threads"])
        self.assertEqual(resources["device"], "cpu")
        self.assertGreaterEqual(cpu_threads, 1)
        self.assertNotIn("cuda_name", resources)

    def test_invalid_precision_is_rejected(self):
        with self.assertRaises(ValueError):
            train_steps(
                make_model(),
                make_batches(),
                steps=1,
                device_name="cpu",
                precision="int8",
            )

    def test_bf16_is_rejected_when_cuda_device_lacks_support(self):
        model = make_model()
        with (
            patch(
                "aviy.training.engine.get_device",
                return_value=torch.device("cuda"),
            ),
            patch("torch.cuda.is_bf16_supported", return_value=False),
            self.assertRaisesRegex(ValueError, "does not support BF16"),
        ):
            train_steps(
                model,
                make_batches(),
                steps=1,
                device_name="cuda",
                precision="bf16",
            )

        self.assertEqual(next(model.parameters()).device.type, "cpu")

    def test_checkpoint_precision_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "precision-checkpoint.pt"
            train_steps(
                make_model(),
                make_batches(),
                steps=1,
                device_name="cpu",
                precision="fp32",
                checkpoint_path=checkpoint,
            )

            with self.assertRaisesRegex(ValueError, "precision does not match"):
                train_steps(
                    make_model(),
                    make_batches(),
                    steps=1,
                    device_name="cpu",
                    precision="auto",
                    resume_from_checkpoint=checkpoint,
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU is not available")
class Phase9CUDATests(unittest.TestCase):
    def test_cuda_training_checkpoint_resume_and_mixed_precision(self):
        precisions = ["fp16"]
        if torch.cuda.is_bf16_supported():
            precisions.append("bf16")

        for precision in precisions:
            with self.subTest(precision=precision):
                torch.cuda.reset_peak_memory_stats()
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "cuda-smoke.pt"
                    model = make_model()
                    with patch(
                        "torch.cuda.reset_peak_memory_stats",
                        wraps=torch.cuda.reset_peak_memory_stats,
                    ) as reset_peak:
                        first = train_steps(
                            model,
                            make_batches(),
                            steps=2,
                            device_name="cuda",
                            seed=42,
                            precision=precision,
                            checkpoint_path=checkpoint,
                        )
                    reset_peak.assert_called_once_with(torch.device("cuda"))
                    self.assertTrue(checkpoint.is_file())

                    resumed = train_steps(
                        model,
                        make_batches(),
                        steps=1,
                        device_name="cuda",
                        seed=42,
                        precision=precision,
                        resume_from_checkpoint=checkpoint,
                        checkpoint_path=checkpoint,
                    )

                    self.assertEqual(first["global_step"], 2)
                    self.assertEqual(resumed["resumed_from_step"], 2)
                    self.assertEqual(resumed["global_step"], 3)
                    resources = cast(dict[str, object], resumed["resources"])
                    first_losses = cast(list[float], first["losses"])
                    resumed_losses = cast(list[float], resumed["losses"])
                    self.assertEqual(resources["device"], "cuda")
                    self.assertIn("cuda_name", resources)
                    self.assertIn("cuda_allocated_bytes", resources)
                    self.assertIn("cuda_reserved_bytes", resources)
                    self.assertIn("cuda_peak_allocated_bytes", resources)
                    self.assertTrue(
                        all(
                            math.isfinite(float(loss))
                            for loss in first_losses + resumed_losses
                        )
                    )


if __name__ == "__main__":
    unittest.main()
