"""Opt-in integration test for real Windows Sandbox agent execution."""

import os
import sys

import pytest

from aviy.agents.runtime import (
    AgentRuntime,
    AuditTrail,
    SandboxToolDefinition,
    ToolRegistry,
)
from aviy.agents.windows_sandbox import WindowsSandboxExecutor

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("AVIY_RUN_WINDOWS_SANDBOX_TESTS") != "1",
        reason="Set AVIY_RUN_WINDOWS_SANDBOX_TESTS=1 to run the real Sandbox test.",
    ),
    pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows Sandbox integration requires Windows.",
    ),
]


def test_agent_permissions_and_real_windows_sandbox_isolation():
    registry = ToolRegistry()
    registry.register(
        SandboxToolDefinition(
            name="sandbox_echo",
            script=r"""
param($Arguments)
[ordered]@{
    received_value = $Arguments.value
    guest_computer = $env:COMPUTERNAME
    guest_identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
}
""",
            required_permission="sandbox.execute",
            timeout_seconds=30,
        )
    )

    executor = WindowsSandboxExecutor()

    # Permission denial must happen before the isolation backend is invoked.
    denied_audit = AuditTrail()
    denied_agent = AgentRuntime(
        "sandbox-denied-agent",
        set(),
        registry,
        denied_audit,
        executor,
    )
    denied_agent.start()

    with pytest.raises(PermissionError, match="required tool permission"):
        denied_agent.invoke("sandbox_echo", {"value": "must not execute"})

    assert [event["event"] for event in denied_audit.events] == [
        "agent_started",
        "tool_denied",
    ]
    assert denied_audit.verify()

    # An authorized agent must execute in the actual disposable guest.
    audit = AuditTrail()
    agent = AgentRuntime(
        "sandbox-authorized-agent",
        {"sandbox.execute"},
        registry,
        audit,
        executor,
    )
    agent.start()

    result = agent.invoke(
        "sandbox_echo",
        {"value": "Aviy integration test"},
    )

    assert result["received_value"] == "Aviy integration test"
    guest_computer = result["guest_computer"]
    guest_identity = result["guest_identity"]
    assert isinstance(guest_computer, str) and guest_computer
    assert isinstance(guest_identity, str) and guest_identity
    assert (
        guest_computer.casefold()
        != __import__("socket").gethostname().casefold()
    )
    assert guest_identity.casefold().endswith(r"\wdagutilityaccount")
    assert [event["event"] for event in audit.events] == [
        "agent_started",
        "tool_started",
        "tool_completed",
    ]
    assert audit.verify()


def test_candidate_experiments_run_inside_real_windows_sandbox(tmp_path):
    from aviy.agents.runtime import SandboxToolDefinition
    from aviy.autonomous.self_improvement import (
        CandidateAdoptionLedger,
        CandidateExperimentRunner,
        ImprovementObjective,
    )

    executor = WindowsSandboxExecutor()
    if executor.isolation_verified is not True:
        pytest.skip("Windows Sandbox is not installed or enabled.")

    baseline = SandboxToolDefinition(
        name="rsi_baseline",
        script=r"""
param($Arguments)
[ordered]@{ accuracy = 0.50 }
""",
        required_permission="sandbox.execute",
        timeout_seconds=45,
    )
    candidate = SandboxToolDefinition(
        name="rsi_candidate",
        script=r"""
param($Arguments)
[ordered]@{ accuracy = 0.80 }
""",
        required_permission="sandbox.execute",
        timeout_seconds=45,
    )
    objective = ImprovementObjective(
        name="accuracy",
        metric="accuracy",
        minimum_gain=0.05,
        maximum_regression=0.0,
        direction="maximize",
        reproducibility_tolerance=0.001,
    )

    runner = CandidateExperimentRunner(tmp_path / "experiments")
    result = runner.evaluate_sandboxed(
        "windows-sandbox-rsi-integration",
        (objective,),
        executor,
        baseline,
        candidate,
        repeats=2,
        configuration={"integration": "real-windows-sandbox"},
    )

    assert result.reproducible is True
    assert result.adopted_eligible is True
    assert result.approval_required is True
    assert result.metric_gains is not None
    assert result.metric_gains["accuracy"] == pytest.approx(0.30)

    ledger = CandidateAdoptionLedger(tmp_path / "adoption.jsonl")
    ledger.record_decision(result, approved=True)
    assert ledger.verify() is True
