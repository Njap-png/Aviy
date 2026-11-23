"""Run a tiny real-CUDA training, checkpoint, and resume smoke test."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from aviy.models.base_model import AviyBaseLM, BaseModelConfig
from aviy.training import train_steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision",
        choices=("auto", "fp16", "bf16", "fp32"),
        default="auto",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/content/aviy_phase9_gpu_smoke.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. In Colab, select Runtime > Change runtime type "
            "> T4 GPU (or another CUDA GPU), then run this script again."
        )
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1.")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit(
            "This GPU does not report BF16 support. Use --precision fp16 or auto."
        )

    torch.manual_seed(42)
    config = BaseModelConfig(
        vocab_size=64,
        context_length=16,
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    )
    model = AviyBaseLM(config)

    generator = torch.Generator().manual_seed(123)
    inputs = torch.randint(0, 64, (2, 16), generator=generator)
    targets = torch.randint(0, 64, (2, 16), generator=generator)
    batches = [(inputs, targets)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_suffix(".checkpoint.pt")

    first = train_steps(
        model,
        batches,
        steps=args.steps,
        device_name="cuda",
        seed=42,
        precision=args.precision,
        checkpoint_path=checkpoint,
    )

    resumed = train_steps(
        model,
        batches,
        steps=1,
        device_name="cuda",
        seed=42,
        precision=args.precision,
        resume_from_checkpoint=checkpoint,
        checkpoint_path=checkpoint,
    )

    losses = list(first["losses"]) + list(resumed["losses"])
    if not all(math.isfinite(float(loss)) for loss in losses):
        raise RuntimeError("Training produced a non-finite loss.")

    resources = resumed["resources"]
    required_resource_fields = {
        "device",
        "cuda_name",
        "cuda_allocated_bytes",
        "cuda_reserved_bytes",
        "cuda_peak_allocated_bytes",
    }
    missing = required_resource_fields.difference(resources)
    if missing:
        raise RuntimeError(f"Missing CUDA resource fields: {sorted(missing)}")

    if resumed["global_step"] != args.steps + 1:
        raise RuntimeError("Checkpoint resume did not restore the expected step.")

    report = {
        "status": "passed",
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "precision": args.precision,
        "initial_training_steps": args.steps,
        "resume_training_steps": 1,
        "global_step_after_resume": resumed["global_step"],
        "losses": losses,
        "resources": resources,
        "checkpoint": str(checkpoint),
    }

    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nGPU smoke test PASSED. Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
