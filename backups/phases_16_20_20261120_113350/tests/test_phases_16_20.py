import hashlib
from pathlib import Path

import pytest

from aviy.agents.runtime import AgentRuntime, AuditTrail, ToolDefinition, ToolRegistry
from aviy.autonomous.self_improvement import (
    CandidateExperimentRunner,
    ImprovementObjective,
)
from aviy.infrastructure.updates import UpdateFile, UpdateManifest, UpdateManager
from aviy.knowledge.research import (
    ResearchEngine,
    ResourceAcquirer,
)
from aviy.models.specialists import (
    MathematicsSpecialist,
    ProgrammingSpecialist,
    ScienceSpecialist,
    SpecialistRouter,
)


def test_specialist_math_is_safe_and_correct():
    specialist = MathematicsSpecialist()
    assert specialist.evaluate_expression("(2 + 3) * 4") == 20
    with pytest.raises(ValueError):
        specialist.evaluate_expression("__import__('os').system('whoami')")


def test_programming_specialist_never_executes_source():
    result = ProgrammingSpecialist().analyze("def add(a, b):\n    return a + b")
    assert result.metadata["valid_syntax"] is True
    assert result.metadata["executed"] is False


def test_specialist_routing_and_science_uncertainty():
    router = SpecialistRouter([
        MathematicsSpecialist(), ProgrammingSpecialist(), ScienceSpecialist()
    ])
    assert router.route("calculate 4 + 5")[0].domain == "mathematics"
    result = ScienceSpecialist().analyze("Does this hypothesis hold?")
    assert result.evidence_required
    assert result.confidence == 0.0


def test_agent_permissions_messaging_and_audit():
    registry = ToolRegistry()
    registry.register(ToolDefinition("echo", lambda args: args, "echo"))
    audit = AuditTrail()
    alice = AgentRuntime("alice", {"echo"}, registry, audit)
    bob = AgentRuntime("bob", set(), registry, audit)
    alice.start()
    bob.start()
    alice.send(bob, "hello")
    assert bob.receive()[0].content == "hello"
    assert alice.invoke("echo", {"value": 3}) == {"value": 3}
    with pytest.raises(PermissionError):
        bob.invoke("echo", {})
    assert len(audit.events) >= 5


def test_agent_tool_arguments_are_copied():
    registry = ToolRegistry()
    registry.register(ToolDefinition("mutate", lambda args: {"seen": args["x"]}, "use"))
    agent = AgentRuntime("agent", {"use"}, registry, AuditTrail())
    agent.start()
    original = {"x": [1, 2]}
    agent.invoke("mutate", original)
    assert original == {"x": [1, 2]}


def test_research_acquisition_requires_approved_host_and_license():
    acquirer = ResourceAcquirer(
        allowed_licenses={"CC-BY-4.0"},
        allowed_hosts={"example.org"},
        maximum_bytes=100,
    )
    source = acquirer.acquire(
        source_id="s1",
        uri="https://example.org/paper",
        title="Example",
        license="CC-BY-4.0",
        content=b"evidence",
    )
    assert source.content == "evidence"
    with pytest.raises(PermissionError):
        acquirer.acquire(
            source_id="s2",
            uri="https://unapproved.org/paper",
            title="Bad",
            license="CC-BY-4.0",
            content=b"x",
        )


def test_research_run_preserves_verification_and_uncertainty():
    from aviy.knowledge.research import ResearchSource

    source = ResearchSource(
        "id", "https://example.org", "title", "CC-BY-4.0",
        "text", hashlib.sha256(b"text").hexdigest(), False, "pending"
    )
    engine = ResearchEngine(lambda query: [source], lambda item: True)
    results = engine.run("research question")
    assert results[0].verified
    assert "corroboration" in results[0].uncertainty


def test_candidate_experiment_requires_reproducible_gain(tmp_path):
    runner = CandidateExperimentRunner(tmp_path)
    result = runner.evaluate(
        "candidate-a",
        (ImprovementObjective("accuracy", "accuracy", 0.05),),
        lambda path: {"accuracy": 0.70},
        lambda path: {"accuracy": 0.80},
        repeats=3,
    )
    assert result.reproducible
    assert result.adopted_eligible
    assert len(result.fingerprint) == 64


def test_candidate_experiment_rejects_regression(tmp_path):
    runner = CandidateExperimentRunner(tmp_path)
    result = runner.evaluate(
        "candidate-b",
        (ImprovementObjective("accuracy", "accuracy", 0.01),),
        lambda path: {"accuracy": 0.8},
        lambda path: {"accuracy": 0.7},
        repeats=2,
    )
    assert not result.adopted_eligible


def _manifest(content: bytes, signature: bytes = b"trusted") -> UpdateManifest:
    return UpdateManifest(
        "update-1",
        "core",
        "1.0",
        (UpdateFile("component.txt", content, hashlib.sha256(content).hexdigest()),),
        signature,
        b"signed manifest",
    )


def test_update_manager_verifies_signature_hash_and_compatibility(tmp_path):
    manager = UpdateManager(
        tmp_path,
        trusted_signature_verifier=lambda payload, signature: signature == b"trusted",
        allowed_components={"core"},
    )
    result = manager.apply(
        _manifest(b"new content"),
        compatibility_check=lambda manifest: True,
    )
    assert not result.rolled_back
    assert (tmp_path / "component.txt").read_bytes() == b"new content"


def test_update_manager_rejects_bad_signature_and_path(tmp_path):
    manager = UpdateManager(
        tmp_path,
        trusted_signature_verifier=lambda payload, signature: signature == b"trusted",
        allowed_components={"core"},
    )
    with pytest.raises(PermissionError):
        manager.verify(_manifest(b"x", b"untrusted"))

    bad = UpdateManifest(
        "update-2", "core", "1.0",
        (UpdateFile("../outside.txt", b"x", hashlib.sha256(b"x").hexdigest()),),
        b"trusted", b"signed",
    )
    with pytest.raises(ValueError):
        manager.verify(bad)


def test_update_manager_rolls_back_partial_application(tmp_path, monkeypatch):
    manager = UpdateManager(
        tmp_path,
        trusted_signature_verifier=lambda payload, signature: True,
        allowed_components={"core"},
    )
    (tmp_path / "one.txt").write_text("old", encoding="utf-8")
    manifest = UpdateManifest(
        "update-3", "core", "1.0",
        (
            UpdateFile("one.txt", b"new", hashlib.sha256(b"new").hexdigest()),
            UpdateFile("two.txt", b"new2", hashlib.sha256(b"new2").hexdigest()),
        ),
        b"signature", b"payload",
    )
    import aviy.infrastructure.updates as updates
    original_replace = updates.os.replace
    calls = {"count": 0}

    def failing_replace(source, destination):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated replacement failure")
        return original_replace(source, destination)

    monkeypatch.setattr(updates.os, "replace", failing_replace)
    with pytest.raises(OSError):
        manager.apply(manifest, compatibility_check=lambda item: True)
    assert (tmp_path / "one.txt").read_text(encoding="utf-8") == "old"
    assert not (tmp_path / "two.txt").exists()
