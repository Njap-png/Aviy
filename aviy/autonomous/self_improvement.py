from __future__ import annotations

import hashlib
import json
import math
import statistics
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from aviy.agents.runtime import SandboxToolDefinition


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
                raise ValueError(f"{field} must be a finite, non-negative number.")


class ExperimentIsolationBackend(Protocol):
    @property
    def isolation_verified(self) -> bool: ...

    def run(
        self,
        experiment: Callable[[Path], dict[str, float]],
        workspace: Path,
        *,
        role: str,
        repeat: int,
    ) -> dict[str, float]: ...


class WindowsSandboxExperimentBackend:
    """Runs RSI baseline and candidate scripts inside disposable Windows Sandboxes.

    The backend deliberately accepts serializable SandboxToolDefinition objects.
    It never executes the supplied Python experiment callbacks on the host.
    """

    def __init__(self, executor, baseline_tool, candidate_tool):
        if not isinstance(baseline_tool, SandboxToolDefinition):
            raise TypeError("Baseline must be a SandboxToolDefinition.")
        if not isinstance(candidate_tool, SandboxToolDefinition):
            raise TypeError("Candidate must be a SandboxToolDefinition.")
        if not callable(getattr(executor, "execute", None)):
            raise TypeError("Executor must provide an execute method.")

        self.executor = executor
        self.baseline_tool = baseline_tool
        self.candidate_tool = candidate_tool

    @property
    def isolation_verified(self) -> bool:
        try:
            return self.executor.isolation_verified is True
        except Exception:  # noqa: BLE001 - isolation checks fail closed for any backend error
            return False

    def run(self, experiment, workspace, *, role, repeat):
        if role not in {"baseline", "candidate"}:
            raise ValueError("Experiment role must be baseline or candidate.")
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
            raise ValueError("Repeat index must be a non-negative integer.")

        # The callback is intentionally not invoked. The serializable PowerShell
        # tool is executed by the OS-isolation backend instead.
        tool = self.baseline_tool if role == "baseline" else self.candidate_tool
        result = self.executor.execute(
            tool,
            {},
            timeout_seconds=tool.timeout_seconds,
        )
        if not isinstance(result, dict):
            raise TypeError("Sandbox experiments must return a metric dictionary.")
        return result


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
        self,
        root: str | Path,
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
                isinstance(metric, str)
                and bool(metric.strip())
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for metric, value in scores.items()
            )
        )

    def evaluate_sandboxed(
        self,
        candidate_id: str,
        objectives: tuple[ImprovementObjective, ...],
        executor,
        baseline_tool: SandboxToolDefinition,
        candidate_tool: SandboxToolDefinition,
        *,
        repeats: int = 3,
        configuration: dict[str, object] | None = None,
    ) -> CandidateResult:
        """Evaluate serializable baseline and candidate tools in real OS sandboxes."""
        backend = WindowsSandboxExperimentBackend(
            executor,
            baseline_tool,
            candidate_tool,
        )
        isolated_runner = CandidateExperimentRunner(self.root, backend)

        def host_execution_forbidden(_workspace):
            raise RuntimeError(
                "Sandboxed experiments must execute through the OS isolation backend."
            )

        return isolated_runner.evaluate(
            candidate_id,
            objectives,
            host_execution_forbidden,
            host_execution_forbidden,
            repeats=repeats,
            configuration=configuration,
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
            raise ValueError(
                "Objectives must be a non-empty tuple of valid objectives."
            )
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
            raise ValueError("At least two integer experiment repeats are required.")
        if not callable(baseline) or not callable(candidate):
            raise TypeError("Baseline and candidate experiments must be callable.")
        if configuration is not None:
            if not isinstance(configuration, dict):
                raise TypeError("Experiment configuration must be a dictionary.")
            try:
                json.dumps(
                    configuration,
                    sort_keys=True,
                    allow_nan=False,
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
                        experiment,
                        directory,
                        role=role,
                        repeat=index,
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


class CandidateAdoptionLedger:
    """Append-only, hash-chained records of explicit candidate decisions.

    Approval records a decision; it does not install, overwrite, or activate code.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _read_records(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []

        records = []
        previous_hash = "0" * 64

        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("Ledger record must be an object.")

                digest = record.pop("hash", None)
                if record.get("sequence") != index:
                    raise ValueError("Invalid ledger sequence.")
                if record.get("previous_hash") != previous_hash:
                    raise ValueError("Invalid ledger hash-chain link.")

                encoded = json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                expected = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
                if not isinstance(digest, str) or digest != expected:
                    raise ValueError("Ledger record hash verification failed.")

                record["hash"] = digest
                records.append(record)
                previous_hash = digest

        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            raise ValueError("Adoption ledger integrity verification failed.")

        return records

    def verify(self) -> bool:
        with self._lock:
            try:
                self._read_records()
                return True
            except ValueError:
                return False

    def record_decision(
        self,
        result: CandidateResult,
        *,
        approved: bool,
    ) -> str:
        if not isinstance(result, CandidateResult):
            raise TypeError("A CandidateResult is required.")
        if type(approved) is not bool:
            raise TypeError("Approval decision must be a boolean.")
        if approved and result.adopted_eligible is not True:
            raise ValueError(
                "An ineligible candidate cannot receive adoption approval."
            )
        if not isinstance(result.fingerprint, str) or len(result.fingerprint) != 64:
            raise ValueError("Candidate fingerprint is invalid.")

        with self._lock:
            records = self._read_records()

            if any(item.get("fingerprint") == result.fingerprint for item in records):
                raise ValueError("A decision already exists for this fingerprint.")

            previous_hash = records[-1]["hash"] if records else "0" * 64
            body = {
                "sequence": len(records),
                "timestamp": time.time(),
                "candidate_id": result.candidate_id,
                "fingerprint": result.fingerprint,
                "eligible": result.adopted_eligible,
                "approved": approved,
                "previous_hash": previous_hash,
            }
            encoded = json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            line = json.dumps(
                {**body, "hash": digest},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )

            existing = (
                self.path.read_text(encoding="utf-8") if self.path.exists() else ""
            )
            if existing and not existing.endswith("\n"):
                existing += "\n"

            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                    dir=self.path.parent,
                    prefix=self.path.name + ".",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary_path = Path(handle.name)
                    handle.write(existing)
                    handle.write(line + "\n")
                    handle.flush()
                    import os

                    os.fsync(handle.fileno())

                temporary_path.replace(self.path)
            except Exception:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                raise

            return digest
