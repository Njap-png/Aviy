"""Standalone, integrity-checked inference runtime for Aviy Base models."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from aviy.inference.generation import GenerationConfig, generate_tokens
from aviy.models.base_model import (
    AviyBaseLM,
    BaseModelConfig,
    validate_tokenizer_compatibility,
)
from aviy.tokenizer.bpe import ByteBPETokenizer

ARTIFACT_FORMAT = "aviy-inference-artifact-v1"
MODEL_FILE = "model.pt"
TOKENIZER_FILE = "tokenizer.json"
MANIFEST_FILE = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid or unreadable JSON file: {path.name}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def save_inference_artifact(
    model: AviyBaseLM,
    tokenizer: ByteBPETokenizer,
    destination: str | Path,
) -> Path:
    """Export model weights and tokenizer without optimizer/training state.

    The destination must not already exist. Files are written to a temporary
    sibling directory and then renamed into place.
    """
    if not isinstance(model, AviyBaseLM):
        raise TypeError("Only AviyBaseLM models can be exported by this runtime")
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise TypeError("Expected an Aviy ByteBPETokenizer")

    model.config.validate()
    validate_tokenizer_compatibility(tokenizer, model.config)

    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"Artifact destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    temporary = Path(tempfile.mkdtemp(prefix=".aviy-inference-", dir=target.parent))
    try:
        weights_path = temporary / MODEL_FILE
        tokenizer_path = temporary / TOKENIZER_FILE

        # Store CPU tensors so the artifact is portable across CPU/CUDA devices.
        cpu_state = {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        }
        torch.save(cpu_state, weights_path)
        tokenizer.save(tokenizer_path)

        manifest = {
            "format": ARTIFACT_FORMAT,
            "model_type": "AviyBaseLM",
            "model_config": asdict(model.config),
            "tokenizer_format": "aviy-byte-bpe-v1",
            "files": {
                MODEL_FILE: _sha256(weights_path),
                TOKENIZER_FILE: _sha256(tokenizer_path),
            },
            "model_initialization": model.config.initialization,
            "pretrained_weights": model.config.pretrained_weights,
        }
        (temporary / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Re-check the written files before publishing the directory.
        for filename, expected_hash in manifest["files"].items():
            if _sha256(temporary / filename) != expected_hash:
                raise RuntimeError(f"Artifact verification failed: {filename}")

        os.replace(temporary, target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class InferenceEngine:
    """Validated text-generation interface for an Aviy Base model."""

    def __init__(
        self,
        model: AviyBaseLM,
        tokenizer: ByteBPETokenizer,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if not isinstance(model, AviyBaseLM):
            raise TypeError("model must be an AviyBaseLM")
        if not isinstance(tokenizer, ByteBPETokenizer):
            raise TypeError("tokenizer must be a ByteBPETokenizer")

        model.config.validate()
        validate_tokenizer_compatibility(tokenizer, model.config)

        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self.model = model.to(resolved_device)
        self.tokenizer = tokenizer
        self.device = resolved_device

    @classmethod
    def from_artifact(
        cls,
        directory: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> InferenceEngine:
        """Load an exported artifact after verifying its integrity hashes."""
        source = Path(directory)
        if not source.is_dir():
            raise FileNotFoundError(f"Inference artifact directory not found: {source}")

        manifest = _read_json(source / MANIFEST_FILE)
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError("Unsupported inference artifact format")
        if manifest.get("model_type") != "AviyBaseLM":
            raise ValueError("Unsupported model type in inference artifact")

        files = manifest.get("files")
        if not isinstance(files, dict):
            raise TypeError("Artifact manifest has no valid file hash map")

        for filename in (MODEL_FILE, TOKENIZER_FILE):
            expected = files.get(filename)
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"Missing or invalid hash for {filename}")
            file_path = source / filename
            if not file_path.is_file():
                raise FileNotFoundError(f"Artifact file is missing: {filename}")
            if _sha256(file_path) != expected:
                raise ValueError(f"Artifact integrity check failed: {filename}")

        raw_config = manifest.get("model_config")
        if not isinstance(raw_config, dict):
            raise TypeError("Artifact model configuration is invalid")
        try:
            config = BaseModelConfig(**raw_config)
            config.validate()
        except (TypeError, ValueError) as exc:
            raise TypeError("Artifact model configuration is invalid") from exc

        tokenizer = ByteBPETokenizer.load(source / TOKENIZER_FILE)
        validate_tokenizer_compatibility(tokenizer, config)

        model = AviyBaseLM(config)
        try:
            state = torch.load(
                source / MODEL_FILE,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise ValueError("Model weights could not be safely loaded") from exc

        if not isinstance(state, dict) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state.items()
        ):
            raise ValueError("Model weights must be a tensor state dictionary")

        try:
            model.load_state_dict(state, strict=True)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(
                "Model weights do not match the declared configuration"
            ) from exc

        return cls(model, tokenizer, device=device)

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate and return only the newly generated text."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt must contain at least one token")
        if len(prompt_ids) > self.model.config.context_length:
            raise ValueError("Prompt exceeds the model context length")

        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self.device,
        )
        generated = generate_tokens(self.model, input_ids, config)
        new_ids = generated[0, len(prompt_ids) :].tolist()
        return self.tokenizer.decode(new_ids)

    def inspect(self) -> dict[str, Any]:
        """Return non-sensitive runtime metadata; no weights or prompt content."""
        return {
            "model_type": "AviyBaseLM",
            "device": str(self.device),
            "context_length": self.model.config.context_length,
            "model_vocabulary_size": self.model.config.vocab_size,
            "tokenizer_vocabulary_size": self.tokenizer.vocabulary_size,
            "parameters": self.model.parameter_count(),
            "initialization": self.model.config.initialization,
            "pretrained_weights": self.model.config.pretrained_weights,
            "training_status": "not_verified_by_inference_runtime",
        }
