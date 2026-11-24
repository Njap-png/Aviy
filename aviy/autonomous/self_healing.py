"""Bounded, auditable fault diagnosis and recovery."""

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fault:
    component: str
    code: str
    message: str
    severity: str = "error"

    def __post_init__(self):
        if not self.component or not self.code or not self.message:
            raise ValueError("Fault fields must be non-empty.")
        if self.severity not in {"warning", "error", "critical"}:
            raise ValueError("Invalid fault severity.")


@dataclass(frozen=True)
class RepairProposal:
    fault_code: str
    action: str
    description: str


@dataclass
class RecoveryReport:
    fault: Fault
    proposed: list = field(default_factory=list)
    applied: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    recovered: bool = False


class FaultDetector:
    def diagnose(self, observations):
        faults = []
        for item in observations:
            if isinstance(item, Fault):
                faults.append(item)
            elif isinstance(item, Mapping):
                faults.append(Fault(**item))
            else:
                raise TypeError("Observations must be Fault objects or mappings.")
        return faults


class RepairPolicy:
    """Only explicitly registered repair actions may execute."""

    def __init__(self, actions, max_actions=2):
        if (
            not isinstance(max_actions, int)
            or isinstance(max_actions, bool)
            or max_actions < 1
        ):
            raise ValueError("max_actions must be a positive integer.")
        if not isinstance(actions, Mapping) or not actions:
            raise ValueError("At least one approved repair action is required.")
        if any(
            not isinstance(k, str) or not k or not callable(v)
            for k, v in actions.items()
        ):
            raise ValueError("Repair actions must be named callables.")
        self._actions = dict(actions)
        self.max_actions = max_actions

    def propose(self, fault, action_names):
        proposals = []
        for name in action_names:
            if name in self._actions and len(proposals) < self.max_actions:
                proposals.append(
                    RepairProposal(fault.code, name, f"Approved repair action: {name}")
                )
        return proposals

    def execute(self, proposal):
        if proposal.action not in self._actions:
            raise PermissionError("Repair action is not approved.")
        return self._actions[proposal.action]()


class RecoveryEngine:
    def __init__(self, detector, policy, verification):
        if not callable(verification):
            raise TypeError("verification must be callable.")
        self.detector = detector
        self.policy = policy
        self.verification = verification

    def recover(self, observations, action_plan):
        reports = []
        for fault in self.detector.diagnose(observations):
            report = RecoveryReport(fault)
            names = action_plan.get(fault.code, ())
            report.proposed = self.policy.propose(fault, names)
            for proposal in report.proposed:
                try:
                    result = self.policy.execute(proposal)
                    report.applied.append(
                        {
                            "action": proposal.action,
                            "result": result,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - recovery must record arbitrary action failures
                    report.errors.append(f"{proposal.action}: {type(exc).__name__}")
                    break
            try:
                report.recovered = bool(self.verification(fault))
            except Exception as exc:  # noqa: BLE001 - verification failures belong in the recovery report
                report.errors.append(f"verification: {type(exc).__name__}")
            reports.append(report)
        return reports
