"""Modality-specific ingestion, independent datasets and evaluation reports."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

SUPPORTED_MODALITIES = ("image", "document", "audio", "speech", "video")


class _FrozenDict(dict):
    """A JSON-compatible dictionary that rejects mutation."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("Sample metadata is immutable.")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    def setdefault(self, key: str, default: object = None) -> object:
        raise TypeError("Sample metadata is immutable.")

    def update(self, *args: object, **kwargs: object) -> None:
        raise TypeError("Sample metadata is immutable.")

    __ior__ = _immutable


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        frozen = _FrozenDict()
        for key, item in value.items():
            dict.__setitem__(frozen, key, _freeze_json(item))
        return frozen
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _json_copy(value: object) -> object:
    try:
        copied = json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
        return copied
    except (TypeError, ValueError) as exc:
        raise ValueError("Metadata and metrics must contain finite JSON values.") from exc


@dataclass(frozen=True)
class ModalitySample:
    sample_id: str
    modality: str
    payload: bytes
    metadata: dict

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string.")
        if not isinstance(self.modality, str) or self.modality not in SUPPORTED_MODALITIES:
            raise ValueError("Unsupported modality.")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("Payload must be non-empty bytes.")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")
        object.__setattr__(self, "metadata", _freeze_json(_json_copy(self.metadata)))


@dataclass(frozen=True)
class ModalityDataset:
    modality: str
    samples: tuple

    def __post_init__(self) -> None:
        if not isinstance(self.modality, str) or self.modality not in SUPPORTED_MODALITIES:
            raise ValueError("Unsupported modality.")
        if not isinstance(self.samples, tuple) or not self.samples:
            raise ValueError("A modality dataset requires a non-empty tuple of samples.")
        if any(not isinstance(sample, ModalitySample) for sample in self.samples):
            raise TypeError("Every dataset entry must be a ModalitySample.")
        if any(sample.modality != self.modality for sample in self.samples):
            raise ValueError("A dataset cannot mix modalities.")
        ids = [sample.sample_id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate sample IDs are not allowed.")

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for sample in self.samples:
            metadata = json.dumps(
                sample.metadata, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ).encode("utf-8")
            for part in (
                self.modality.encode("utf-8"),
                sample.sample_id.encode("utf-8"),
                sample.payload,
                metadata,
            ):
                digest.update(len(part).to_bytes(8, "big"))
                digest.update(part)
        return digest.hexdigest()


@dataclass(frozen=True)
class EvaluationReport:
    modality: str
    dataset_fingerprint: str
    sample_count: int
    metrics: dict


class ModalityInterface:
    def __init__(
        self, modality: str, processor: Callable,
        max_payload_bytes: int = 20_000_000,
    ) -> None:
        if not isinstance(modality, str) or modality not in SUPPORTED_MODALITIES:
            raise ValueError("A supported modality is required.")
        if not callable(processor):
            raise TypeError("processor must be callable.")
        if (
            isinstance(max_payload_bytes, bool)
            or not isinstance(max_payload_bytes, int)
            or max_payload_bytes < 1
        ):
            raise ValueError("Invalid payload limit.")
        self.modality = modality
        self.processor = processor
        self.max_payload_bytes = max_payload_bytes

    def ingest(
        self, sample_id: str, payload: bytes, metadata: dict | None = None
    ) -> ModalitySample:
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary.")
        sample = ModalitySample(sample_id, self.modality, payload, metadata)
        if len(sample.payload) > self.max_payload_bytes:
            raise ValueError("Payload exceeds modality size limit.")
        return sample

    def process(self, sample: ModalitySample):
        if not isinstance(sample, ModalitySample) or sample.modality != self.modality:
            raise ValueError("Sample does not belong to this modality interface.")
        return self.processor(sample.payload, copy.deepcopy(sample.metadata))


class MultimodalWorkbench:
    def __init__(self, interfaces: Mapping[str, ModalityInterface]) -> None:
        if not isinstance(interfaces, Mapping):
            raise TypeError("interfaces must be a mapping.")
        self.interfaces = dict(interfaces)
        if set(self.interfaces) != set(SUPPORTED_MODALITIES):
            raise ValueError("Provide separate interfaces for every supported modality.")
        for name, interface in self.interfaces.items():
            if not isinstance(interface, ModalityInterface) or interface.modality != name:
                raise ValueError("Interface key and modality do not match.")

    def build_dataset(self, modality: str, samples) -> ModalityDataset:
        if modality not in self.interfaces:
            raise ValueError("Unsupported modality.")
        return ModalityDataset(modality, tuple(samples))

    def evaluate(self, dataset: ModalityDataset, evaluator: Callable) -> EvaluationReport:
        if not isinstance(dataset, ModalityDataset) or not callable(evaluator):
            raise TypeError("Dataset and evaluator are required.")
        if dataset.modality not in self.interfaces:
            raise ValueError("No interface registered for dataset modality.")
        results = [
            self.interfaces[dataset.modality].process(sample)
            for sample in dataset.samples
        ]
        metrics = evaluator(results, dataset)
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("Evaluator must return a non-empty metrics dictionary.")
        safe_metrics = _json_copy(metrics)
        if not isinstance(safe_metrics, dict):
            raise TypeError("Evaluator metrics must serialize to a dictionary.")
        return EvaluationReport(
            dataset.modality, dataset.fingerprint, len(dataset.samples), safe_metrics
        )
