"""Reproducible architecture experiments with isolated workspaces and gated selection."""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


@dataclass(frozen=True)
class ArchitectureCandidate:
    name: str
    configuration: Mapping

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Candidate name is required.")
        if not isinstance(self.configuration, Mapping):
            raise TypeError("Candidate configuration must be a mapping.")
        json.dumps(dict(self.configuration), sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class ExperimentResult:
    candidate: str
    fingerprint: str
    metrics: Mapping
    reproducible: bool
    improved: bool
    selected: bool


class ArchitectureSearchSpace:
    def __init__(self, candidates):
        self.candidates = tuple(candidates)
        if not self.candidates:
            raise ValueError("Search space cannot be empty.")
        if any(not isinstance(c, ArchitectureCandidate) for c in self.candidates):
            raise TypeError("Search space accepts ArchitectureCandidate objects only.")
        names = [c.name for c in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("Candidate names must be unique.")

    @classmethod
    def from_registry(cls, registry, configurations):
        names = set(registry.names)
        unknown = set(configurations) - names
        if unknown:
            raise ValueError(f"Unknown registered architectures: {sorted(unknown)}")
        return cls(
            [
                ArchitectureCandidate(name, config)
                for name, config in configurations.items()
            ]
        )


class ArchitectureExperimentRunner:
    """Evaluators receive a fresh temporary workspace for every run."""

    def __init__(self, evaluator, metric, minimum_gain=0.0):
        if not callable(evaluator) or not isinstance(metric, str) or not metric:
            raise ValueError("A callable evaluator and metric name are required.")
        if not math.isfinite(minimum_gain) or minimum_gain < 0:
            raise ValueError("minimum_gain must be finite and non-negative.")
        self.evaluator = evaluator
        self.metric = metric
        self.minimum_gain = minimum_gain

    @staticmethod
    def _metrics(value):
        if not isinstance(value, Mapping) or not value:
            raise ValueError("Evaluator must return a non-empty metric mapping.")
        result = dict(value)
        for key, number in result.items():
            if not isinstance(key, str) or not math.isfinite(float(number)):
                raise ValueError("Metrics must have string names and finite values.")
            result[key] = float(number)
        return result

    @staticmethod
    def _fingerprint(candidate):
        payload = json.dumps(
            {"name": candidate.name, "configuration": dict(candidate.configuration)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _run_once(self, candidate):
        with TemporaryDirectory(prefix="aviy-architecture-") as directory:
            workspace = Path(directory)
            return self._metrics(self.evaluator(candidate, workspace))

    def evaluate(self, candidate, baseline):
        if not isinstance(candidate, ArchitectureCandidate):
            raise TypeError("candidate must be an ArchitectureCandidate.")
        baseline = self._metrics(baseline)
        first = self._run_once(candidate)
        second = self._run_once(candidate)
        reproducible = first == second
        if self.metric not in baseline or self.metric not in first:
            raise ValueError(f"Missing comparison metric: {self.metric}")
        improved = reproducible and (
            first[self.metric] >= baseline[self.metric] + self.minimum_gain
            and first[self.metric] > baseline[self.metric]
        )
        return ExperimentResult(
            candidate.name,
            self._fingerprint(candidate),
            first,
            reproducible,
            improved,
            improved,
        )

    def search(self, space, baseline):
        if not isinstance(space, ArchitectureSearchSpace):
            raise TypeError("space must be an ArchitectureSearchSpace.")
        return [self.evaluate(candidate, baseline) for candidate in space.candidates]
