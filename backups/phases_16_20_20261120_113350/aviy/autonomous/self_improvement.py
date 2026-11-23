from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ImprovementObjective:
    name: str
    metric: str
    minimum_gain: float
    maximum_regression: float = 0.0


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    baseline_scores: dict[str, tuple[float, ...]]
    candidate_scores: dict[str, tuple[float, ...]]
    reproducible: bool
    adopted_eligible: bool
    reasons: tuple[str, ...]
    fingerprint: str


class CandidateExperimentRunner:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _valid_scores(scores: dict[str, float]) -> bool:
        return bool(scores) and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in scores.values()
        )

    def evaluate(
        self,
        candidate_id: str,
        objectives: tuple[ImprovementObjective, ...],
        baseline: Callable[[Path], dict[str, float]],
        candidate: Callable[[Path], dict[str, float]],
        *,
        repeats: int = 3,
    ) -> CandidateResult:
        if not candidate_id.strip() or not objectives or repeats < 2:
            raise ValueError("Candidate, objectives and at least two repeats required.")
        if len({item.metric for item in objectives}) != len(objectives):
            raise ValueError("Objective metrics must be unique.")

        baseline_scores: dict[str, list[float]] = {}
        candidate_scores: dict[str, list[float]] = {}
        reasons: list[str] = []

        with tempfile.TemporaryDirectory(
            prefix="aviy-candidate-", dir=self.root
        ) as temporary:
            workspace = Path(temporary)
            for index in range(repeats):
                base_dir = workspace / f"baseline-{index}"
                candidate_dir = workspace / f"candidate-{index}"
                base_dir.mkdir()
                candidate_dir.mkdir()
                base = baseline(base_dir)
                trial = candidate(candidate_dir)
                if not self._valid_scores(base) or not self._valid_scores(trial):
                    raise ValueError("Experiment returned invalid metric scores.")
                for metric, value in base.items():
                    baseline_scores.setdefault(metric, []).append(float(value))
                for metric, value in trial.items():
                    candidate_scores.setdefault(metric, []).append(float(value))

        reproducible = True
        eligible = True
        for objective in objectives:
            base = baseline_scores.get(objective.metric, [])
            trial = candidate_scores.get(objective.metric, [])
            if len(base) != repeats or len(trial) != repeats:
                eligible = False
                reasons.append(f"Missing metric: {objective.metric}")
                continue
            if max(base) - min(base) > 1e-9 or max(trial) - min(trial) > 1e-9:
                reproducible = False
                eligible = False
                reasons.append(f"Non-reproducible metric: {objective.metric}")
                continue
            gain = sum(trial) / repeats - sum(base) / repeats
            if gain < objective.minimum_gain:
                eligible = False
                reasons.append(f"Insufficient gain: {objective.metric}")
            if gain < -objective.maximum_regression:
                eligible = False
                reasons.append(f"Regression: {objective.metric}")

        payload = {
            "candidate_id": candidate_id,
            "baseline": baseline_scores,
            "candidate": candidate_scores,
            "reproducible": reproducible,
            "eligible": eligible,
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        return CandidateResult(
            candidate_id,
            {key: tuple(value) for key, value in baseline_scores.items()},
            {key: tuple(value) for key, value in candidate_scores.items()},
            reproducible,
            eligible and reproducible,
            tuple(reasons),
            fingerprint,
        )
