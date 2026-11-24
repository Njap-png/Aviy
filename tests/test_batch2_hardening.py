import json
import math
import threading
from typing import Any, cast

import pytest

from aviy.multimodal.pipelines import (
    SUPPORTED_MODALITIES,
    ModalityDataset,
    ModalityInterface,
    ModalitySample,
    MultimodalWorkbench,
)
from aviy.research.programs import compare_benchmarks
from aviy.security.durable_governance import (
    DurableGovernanceLedger,
    GovernedRecovery,
    LedgerIntegrityError,
)
from aviy.security.governance import (
    DualApproval,
    EnvironmentSecretResolver,
    FilesystemBoundary,
    GovernanceLedger,
    NetworkPolicy,
    Principal,
)

KEY = b"batch-two-governance-integrity-key"


def _workbench():
    return MultimodalWorkbench({
        name: ModalityInterface(name, lambda payload, metadata: len(payload))
        for name in SUPPORTED_MODALITIES
    })


def test_multimodal_fingerprint_uses_unambiguous_boundaries():
    a = ModalityDataset("image", (
        ModalitySample("ab", "image", b"c", {}),
    ))
    b = ModalityDataset("image", (
        ModalitySample("a", "image", b"bc", {}),
    ))
    assert a.fingerprint != b.fingerprint


def test_multimodal_fingerprint_includes_metadata():
    a = ModalityDataset("image", (
        ModalitySample("id", "image", b"x", {"source": "one"}),
    ))
    b = ModalityDataset("image", (
        ModalitySample("id", "image", b"x", {"source": "two"}),
    ))
    assert a.fingerprint != b.fingerprint


def test_multimodal_rejects_invalid_ids_and_metadata():
    interface = ModalityInterface("image", lambda payload, metadata: True)
    with pytest.raises(ValueError):
        interface.ingest("", b"x")
    with pytest.raises(TypeError):
        interface.ingest("id", b"x", cast(Any, []))
    with pytest.raises(ValueError):
        interface.ingest("id", b"x", {"score": float("nan")})


def test_multimodal_rejects_boolean_payload_limit():
    with pytest.raises(ValueError):
        ModalityInterface("image", lambda p, m: True, True)


def test_multimodal_rejects_wrong_interface_types():
    interfaces = {name: ModalityInterface(name, lambda p, m: True)
                  for name in SUPPORTED_MODALITIES}
    interfaces["image"] = cast(Any, object())
    with pytest.raises(ValueError):
        MultimodalWorkbench(interfaces)


def test_benchmark_output_cannot_overwrite_input(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text('{"metrics":{"score":1}}', encoding="utf-8")
    candidate.write_text('{"metrics":{"score":2}}', encoding="utf-8")
    original = baseline.read_bytes()
    with pytest.raises(ValueError):
        compare_benchmarks(baseline, candidate, baseline)
    assert baseline.read_bytes() == original


def test_benchmark_rejects_non_finite_json(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text('{"metrics":{"score":NaN}}', encoding="utf-8")
    candidate.write_text('{"metrics":{"score":2}}', encoding="utf-8")
    with pytest.raises(ValueError):
        compare_benchmarks(baseline, candidate, tmp_path / "report.json")


def test_benchmark_does_not_treat_booleans_as_metrics(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text('{"metrics":{"flag":true}}', encoding="utf-8")
    candidate.write_text('{"metrics":{"flag":false}}', encoding="utf-8")
    report = compare_benchmarks(baseline, candidate, tmp_path / "report.json")
    assert "flag" not in report["comparisons"]


def test_durable_ledger_rejects_non_finite_details(tmp_path):
    ledger = DurableGovernanceLedger(tmp_path / "ledger.jsonl", KEY)
    with pytest.raises(ValueError):
        ledger.append("system", "test", {"value": math.nan})


def test_durable_ledger_instances_share_thread_lock(tmp_path):
    path = tmp_path / "ledger.jsonl"
    first = DurableGovernanceLedger(path, KEY)
    second = DurableGovernanceLedger(path, KEY)
    errors = []

    def append_many(ledger, prefix):
        try:
            for index in range(20):
                ledger.append(prefix, "event", {"index": index})
        except (OSError, ValueError, TypeError, LedgerIntegrityError) as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=append_many, args=(first, "first")),
        threading.Thread(target=append_many, args=(second, "second")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert first.verify()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 40
    assert [record["sequence"] for record in records] == list(range(1, 41))


def test_recovery_rejects_string_as_roles(tmp_path):
    ledger = DurableGovernanceLedger(tmp_path / "ledger.jsonl", KEY)
    recovery = GovernedRecovery(ledger)
    with pytest.raises(ValueError):
        recovery.run(
            "proposal",
            [("alice", "recovery_approver"), ("bob", ["recovery_approver"])],
            lambda: True, lambda: True, lambda: True,
        )


def test_recovery_rejects_malformed_approval_record(tmp_path):
    ledger = DurableGovernanceLedger(tmp_path / "ledger.jsonl", KEY)
    with pytest.raises(ValueError):
        GovernedRecovery(ledger).run(
            "proposal", cast(Any, [("alice", ["recovery_approver"]), ("bob",)]),
            lambda: True, lambda: True, lambda: True,
        )


def test_principal_rejects_invalid_roles():
    with pytest.raises(ValueError):
        Principal("alice", frozenset({"approver", 42}))


def test_dual_approval_requires_approver_role():
    approval = DualApproval()
    with pytest.raises(PermissionError):
        approval.approve([
            Principal("alice", frozenset({"reader"})),
            Principal("bob", frozenset({"reader"})),
        ])


def test_environment_secret_name_must_be_ascii_identifier():
    resolver = EnvironmentSecretResolver({"1SECRET": "value"})
    with pytest.raises(ValueError):
        resolver.resolve("1SECRET")


def test_network_policy_rejects_malformed_ipv6_url():
    policy = NetworkPolicy({"example.org"})
    with pytest.raises(PermissionError):
        policy.validate("https://[broken")


def test_filesystem_boundary_rejects_windows_drive_relative_paths(tmp_path):
    boundary = FilesystemBoundary(tmp_path / "safe")
    with pytest.raises(PermissionError):
        boundary.validate("C:outside.txt")


def test_governance_ledger_rejects_non_json_evidence():
    ledger = GovernanceLedger()
    with pytest.raises(ValueError):
        ledger.append("alice", "deploy", "approved", {"score": float("nan")})


def test_governance_ledger_entries_are_defensive_copies():
    ledger = GovernanceLedger()
    ledger.append("alice", "deploy", "approved", {"nested": {"value": 1}})
    exposed = ledger.entries
    exposed[0]["payload"]["evidence"]["nested"]["value"] = 99
    assert ledger.verify()
    assert ledger.entries[0]["payload"]["evidence"]["nested"]["value"] == 1
