import pytest

from aviy.autonomous.architecture_evolution import (
    ArchitectureCandidate,
    ArchitectureExperimentRunner,
    ArchitectureSearchSpace,
)
from aviy.autonomous.self_healing import (
    Fault,
    FaultDetector,
    RecoveryEngine,
    RepairPolicy,
)
from aviy.infrastructure.resources import (
    ResourceMonitor,
    ResourceQuota,
    ResourceSnapshot,
    WorkloadScheduler,
)
from aviy.multimodal.pipelines import (
    SUPPORTED_MODALITIES,
    ModalityInterface,
    MultimodalWorkbench,
)
from aviy.security.governance import (
    AuthorizationPolicy,
    DualApproval,
    EnvironmentSecretResolver,
    FilesystemBoundary,
    GovernanceLedger,
    NetworkPolicy,
    Principal,
)


def test_fault_detection_and_recovery():
    state = {"healthy": False}
    policy = RepairPolicy(
        {"restart": lambda: state.update(healthy=True)}, max_actions=1
    )
    engine = RecoveryEngine(FaultDetector(), policy, lambda fault: state["healthy"])
    reports = engine.recover(
        [Fault("worker", "worker_down", "Worker stopped")],
        {"worker_down": ["restart", "unapproved"]},
    )
    assert reports[0].recovered
    assert len(reports[0].proposed) == 1
    assert reports[0].applied[0]["action"] == "restart"


def test_fault_isolation_and_failure_does_not_stop_other_faults():
    calls = []
    policy = RepairPolicy({"safe": lambda: calls.append("safe")})
    engine = RecoveryEngine(FaultDetector(), policy, lambda f: f.code == "second")
    reports = engine.recover(
        [Fault("a", "first", "failure"), Fault("b", "second", "failure")],
        {"first": ["unknown"], "second": ["safe"]},
    )
    assert len(reports) == 2
    assert not reports[0].recovered
    assert reports[1].recovered
    assert calls == ["safe"]


def test_repair_policy_rejects_invalid_limits_and_unapproved_actions():
    with pytest.raises(ValueError):
        RepairPolicy({"ok": lambda: None}, 0)
    policy = RepairPolicy({"ok": lambda: None})
    with pytest.raises(PermissionError):
        policy.execute(type("P", (), {"action": "dangerous"})())


def test_fault_rejects_invalid_severity():
    with pytest.raises(ValueError):
        Fault("worker", "bad", "message", "unknown")


def test_architecture_search_is_reproducible_and_gated(tmp_path):
    def evaluator(candidate, workspace):
        (workspace / "experiment.txt").write_text(candidate.name)
        return {"accuracy": 0.9}

    candidate = ArchitectureCandidate("candidate_a", {"layers": 2})
    runner = ArchitectureExperimentRunner(evaluator, "accuracy", 0.05)
    result = runner.evaluate(candidate, {"accuracy": 0.8})
    assert result.reproducible and result.improved and result.selected
    assert len(result.fingerprint) == 64


def test_architecture_search_rejects_non_improvement():
    candidate = ArchitectureCandidate("candidate_b", {"layers": 1})
    runner = ArchitectureExperimentRunner(lambda c, w: {"score": 0.5}, "score")
    result = runner.evaluate(candidate, {"score": 0.5})
    assert result.reproducible and not result.improved and not result.selected


def test_architecture_search_rejects_non_reproducible_candidate():
    counter = {"n": 0}

    def evaluator(candidate, workspace):
        counter["n"] += 1
        return {"score": counter["n"]}

    result = ArchitectureExperimentRunner(evaluator, "score").evaluate(
        ArchitectureCandidate("unstable", {}), {"score": 0}
    )
    assert not result.reproducible and not result.selected


def test_search_space_rejects_duplicate_names():
    candidate = ArchitectureCandidate("same", {})
    with pytest.raises(ValueError):
        ArchitectureSearchSpace([candidate, candidate])


def make_workbench():
    interfaces = {
        name: ModalityInterface(name, lambda payload, metadata: len(payload))
        for name in SUPPORTED_MODALITIES
    }
    return MultimodalWorkbench(interfaces)


@pytest.mark.parametrize("modality", SUPPORTED_MODALITIES)
def test_each_modality_has_independent_dataset_and_report(modality):
    workbench = make_workbench()
    interface = workbench.interfaces[modality]
    sample = interface.ingest(f"{modality}-1", b"sample")
    dataset = workbench.build_dataset(modality, [sample])
    report = workbench.evaluate(
        dataset, lambda results, ds: {"processed": len(results)}
    )
    assert report.modality == modality
    assert report.sample_count == 1
    assert report.dataset_fingerprint == dataset.fingerprint


def test_multimodal_workbench_requires_all_supported_interfaces():
    with pytest.raises(ValueError):
        MultimodalWorkbench({})


def test_modality_dataset_rejects_mixed_modalities():
    workbench = make_workbench()
    image = workbench.interfaces["image"].ingest("image-1", b"x")
    audio = workbench.interfaces["audio"].ingest("audio-1", b"x")
    with pytest.raises(ValueError):
        workbench.build_dataset("image", [image, audio])


def test_modality_payload_limit_is_enforced():
    interface = ModalityInterface("image", lambda p, m: True, max_payload_bytes=2)
    with pytest.raises(ValueError):
        interface.ingest("large", b"too large")


def snapshot(cpu=10, gpu=100, ram=1000, free=10000):
    return ResourceSnapshot(cpu, gpu, ram, free)


def test_resource_scheduler_accepts_and_runs_within_quota():
    monitor = ResourceMonitor(lambda: snapshot())
    scheduler = WorkloadScheduler(ResourceQuota(), monitor)
    assert scheduler.submit("job-1", lambda: 42)["accepted"]
    assert scheduler.run_next()["result"] == 42
    assert scheduler.report()["completed"] == 1


def test_resource_scheduler_rejects_cpu_quota():
    scheduler = WorkloadScheduler(
        ResourceQuota(max_cpu_percent=50),
        ResourceMonitor(lambda: snapshot(cpu=90)),
    )
    result = scheduler.submit("job", lambda: None)
    assert not result["accepted"] and "cpu_quota" in result["reasons"]


def test_resource_scheduler_enforces_queue_limit_and_duplicate_ids():
    scheduler = WorkloadScheduler(
        ResourceQuota(max_queued_jobs=1),
        ResourceMonitor(lambda: snapshot()),
    )
    assert scheduler.submit("one", lambda: 1)["accepted"]
    with pytest.raises(ValueError):
        scheduler.submit("one", lambda: 2)
    assert not scheduler.submit("two", lambda: 2)["accepted"]


def test_resource_scheduler_isolates_job_failures():
    scheduler = WorkloadScheduler(ResourceQuota(), ResourceMonitor(lambda: snapshot()))
    scheduler.submit("bad", lambda: 1 / 0)
    scheduler.submit("good", lambda: "ok")
    assert scheduler.run_next()["status"] == "failed"
    assert scheduler.run_next()["result"] == "ok"


def test_resource_snapshot_rejects_invalid_values():
    with pytest.raises(ValueError):
        snapshot(cpu=101)


def test_authorization_policy_enforces_roles():
    policy = AuthorizationPolicy({"admin": {"system.write"}, "reader": {"system.read"}})
    reader = Principal("alice", frozenset({"reader"}))
    assert policy.authorize(reader, "system.read")
    with pytest.raises(PermissionError):
        policy.require(reader, "system.write")


def test_environment_secret_resolver_never_requires_plaintext_config():
    resolver = EnvironmentSecretResolver({"AVIY_TEST_SECRET": "secret-value"})
    assert resolver.resolve("AVIY_TEST_SECRET") == "secret-value"
    with pytest.raises(KeyError):
        resolver.resolve("MISSING_SECRET")


def test_network_policy_restricts_hosts_schemes_and_ports():
    policy = NetworkPolicy({"example.org"})
    assert policy.validate("https://example.org/resource")
    for url in (
        "http://example.org",
        "https://evil.example",
        "https://user:pass@example.org",
        "https://example.org:444",
        "https://127.0.0.1",
    ):
        with pytest.raises(PermissionError):
            policy.validate(url)


def test_filesystem_boundary_rejects_traversal_and_symlinks(tmp_path, monkeypatch):
    boundary = FilesystemBoundary(tmp_path / "safe")
    assert (
        boundary.validate("data/file.txt").parent
        == (tmp_path / "safe" / "data").resolve()
    )
    with pytest.raises(PermissionError):
        boundary.validate("../outside")

    # Simulate an existing symlink without requiring Windows developer mode
    # or elevated privileges to create an actual filesystem symlink.
    path_type = type(boundary.root)
    original_is_symlink = path_type.is_symlink

    def simulated_is_symlink(candidate):
        if candidate == boundary.root / "link":
            return True
        return original_is_symlink(candidate)

    monkeypatch.setattr(path_type, "is_symlink", simulated_is_symlink)
    with pytest.raises(PermissionError, match="Symlinks are forbidden"):
        boundary.validate("link/file.txt")


def test_governance_ledger_detects_tampering():
    ledger = GovernanceLedger()
    ledger.append("alice", "deploy", "approved", {"ticket": "A-1"})
    ledger.append("bob", "deploy", "verified")
    assert ledger.verify()
    entries = ledger._entries
    entries[0]["payload"]["decision"] = "rejected"
    assert not ledger.verify()


def test_dual_approval_requires_two_distinct_people():
    alice = Principal("alice", frozenset({"approver"}))
    bob = Principal("bob", frozenset({"approver"}))
    approval = DualApproval()
    with pytest.raises(PermissionError):
        approval.approve([alice, alice])
    assert approval.approve([alice, bob])
