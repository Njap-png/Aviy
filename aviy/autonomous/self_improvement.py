from __future__ import annotations

import hashlib
import json
import math
import statistics
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ImprovementObjective:
    name: str
    metric: str
    minimum_gain: float
    maximum_regression: float = 0.0
    direction: str = "maximize"
    reproducibility_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Objective name must be a non-empty string.")
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise ValueError("Objective metric must be a non-empty string.")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("Metric direction must be maximize or minimize.")

        thresholds = {
            "minimum_gain": self.minimum_gain,
            "maximum_regression": self.maximum_regression,
            "reproducibility_tolerance": self.reproducibility_tolerance,
        }
        for field, value in thresholds.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"{field} must be a finite, non-negative number."
                )


class ExperimentIsolationBackend(Protocol):
    @property
    def isolation_verified(self) -> bool: ...

    def run(
        self, experiment: Callable[[Path], dict[str, float]], workspace: Path,
        *, role: str, repeat: int,
    ) -> dict[str, float]: ...


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    baseline_scores: dict[str, tuple[float, ...]]
    candidate_scores: dict[str, tuple[float, ...]]
    reproducible: bool
    adopted_eligible: bool
    reasons: tuple[str, ...]
    fingerprint: str
    metric_gains: dict[str, float] | None = None
    baseline_stddev: dict[str, float] | None = None
    candidate_stddev: dict[str, float] | None = None
    approval_required: bool = True


class CandidateExperimentRunner:
    def __init__(
        self, root: str | Path,
        isolation_backend: ExperimentIsolationBackend | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.isolation_backend = isolation_backend

    @staticmethod
    def _valid_scores(scores: dict[str, float]) -> bool:
        return (
            isinstance(scores, dict)
            and bool(scores)
            and all(
                isinstance(metric, str) and bool(metric.strip())
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for metric, value in scores.items()
            )
        )

    def evaluate(
        self,
        candidate_id: str,
        objectives: tuple[ImprovementObjective, ...],
        baseline: Callable[[Path], dict[str, float]],
        candidate: Callable[[Path], dict[str, float]],
        *,
        repeats: int = 3,
        configuration: dict[str, object] | None = None,
    ) -> CandidateResult:
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("Candidate ID must be a non-empty string.")
        if (
            not isinstance(objectives, tuple)
            or not objectives
            or any(not isinstance(item, ImprovementObjective) for item in objectives)
        ):
            raise ValueError("Objectives must be a non-empty tuple of valid objectives.")
        if (
            isinstance(repeats, bool)
            or not isinstance(repeats, int)
            or repeats < 2
        ):
            raise ValueError("At least two integer experiment repeats are required.")
        if not callable(baseline) or not callable(candidate):
            raise TypeError("Baseline and candidate experiments must be callable.")
        if configuration is not None:
            if not isinstance(configuration, dict):
                raise TypeError("Experiment configuration must be a dictionary.")
            try:
                json.dumps(
                    configuration, sort_keys=True, allow_nan=False,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Experiment configuration must contain finite JSON values."
                ) from exc
        if len({item.metric for item in objectives}) != len(objectives):
            raise ValueError("Objective metrics must be unique.")
        backend = self.isolation_backend
        if backend is None or getattr(backend, "isolation_verified", None) is not True:
            raise RuntimeError(
                "Candidate execution blocked: a verified experiment-isolation "
                "backend is required."
            )

        baseline_scores: dict[str, list[float]] = {}
        candidate_scores: dict[str, list[float]] = {}
        reasons: list[str] = []
        with tempfile.TemporaryDirectory(
            prefix="aviy-candidate-", dir=self.root
        ) as temporary:
            workspace = Path(temporary)
            for index in range(repeats):
                for role, experiment, target in (
                    ("baseline", baseline, baseline_scores),
                    ("candidate", candidate, candidate_scores),
                ):
                    directory = workspace / f"{role}-{index}"
                    directory.mkdir()
                    scores = backend.run(
                        experiment, directory, role=role, repeat=index,
                    )
                    if not self._valid_scores(scores):
                        raise ValueError("Experiment returned invalid metric scores.")
                    if any(
                        metric not in {item.metric for item in objectives}
                        for metric in scores
                    ):
                        raise ValueError(
                            "Experiment returned a metric not declared by its objectives."
                        )
                    for metric, value in scores.items():
                        target.setdefault(metric, []).append(float(value))

        reproducible = True
        eligible = True
        gains: dict[str, float] = {}
        base_std: dict[str, float] = {}
        candidate_std: dict[str, float] = {}
        for objective in objectives:
            metric = objective.metric
            base = baseline_scores.get(metric, [])
            trial = candidate_scores.get(metric, [])
            if len(base) != repeats or len(trial) != repeats:
                eligible = False
                reasons.append(f"Missing metric or incomplete repeats: {metric}")
                continue
            base_std[metric] = statistics.pstdev(base)
            candidate_std[metric] = statistics.pstdev(trial)
            tolerance = objective.reproducibility_tolerance
            if max(base) - min(base) > tolerance or max(trial) - min(trial) > tolerance:
                reproducible = False
                eligible = False
                reasons.append(f"Non-reproducible metric: {metric}")
            raw_gain = statistics.mean(trial) - statistics.mean(base)
            gain = raw_gain if objective.direction == "maximize" else -raw_gain
            gains[metric] = gain
            if gain < objective.minimum_gain:
                eligible = False
                reasons.append(f"Insufficient improvement: {metric}")
            if gain < -objective.maximum_regression:
                eligible = False
                reasons.append(f"Regression limit exceeded: {metric}")

        payload = {
            "candidate_id": candidate_id,
            "configuration": configuration or {},
            "baseline": baseline_scores,
            "candidate": candidate_scores,
            "objectives": [item.__dict__ for item in objectives],
            "gains": gains,
            "reproducible": reproducible,
            "eligible": eligible,
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return CandidateResult(
            candidate_id,
            {key: tuple(value) for key, value in baseline_scores.items()},
            {key: tuple(value) for key, value in candidate_scores.items()},
            reproducible,
            eligible and reproducible,
            tuple(reasons),
            fingerprint,
            gains,
            base_std,
            candidate_std,
            True,
        )
