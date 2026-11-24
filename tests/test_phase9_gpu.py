"""CPU-safe and CUDA-conditional tests for Phase 9 training infrastructure."""

import math
import tempfile
import unittest
from pathlib import Path
from typing import cast

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
                    first = train_steps(
                        model,
                        make_batches(),
                        steps=2,
                        device_name="cuda",
                        seed=42,
                        precision=precision,
                        checkpoint_path=checkpoint,
                    )
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
