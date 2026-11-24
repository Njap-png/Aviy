import json

import pytest

from aviy.security.authentication import (
    AuthenticationError,
    AuthenticationService,
    AuthorizationError,
)
from aviy.security.durable_governance import (
    DurableGovernanceLedger,
    GovernedRecovery,
    LedgerIntegrityError,
    RecoveryVerificationError,
)

KEY = b"phase25-test-signing-key-32-bytes-minimum!!"


def make_auth(tmp_path, clock=None):
    kwargs = {"clock": clock} if clock else {}
    return AuthenticationService(tmp_path / "auth.sqlite3", KEY, **kwargs)


def test_passwords_are_persisted_as_hashes_and_login_works(tmp_path):
    auth = make_auth(tmp_path)
    auth.register("Alice", "a-long-unique-password", ["admin"])
    token = auth.authenticate("ALICE", "a-long-unique-password")
    assert auth.verify(token).username == "alice"
    raw = (tmp_path / "auth.sqlite3").read_bytes()
    assert b"a-long-unique-password" not in raw


def test_duplicate_accounts_and_weak_passwords_are_rejected(tmp_path):
    auth = make_auth(tmp_path)
    with pytest.raises(ValueError):
        auth.register("alice", "short")
    auth.register("alice", "a-long-unique-password")
    with pytest.raises(ValueError):
        auth.register("ALICE", "another-long-password")


def test_bad_credentials_do_not_authenticate(tmp_path):
    auth = make_auth(tmp_path)
    auth.register("alice", "a-long-unique-password")
    with pytest.raises(AuthenticationError):
        auth.authenticate("alice", "incorrect-password")
    with pytest.raises(AuthenticationError):
        auth.authenticate("unknown", "incorrect-password")


def test_roles_are_checked_and_updates_take_effect_immediately(tmp_path):
    auth = make_auth(tmp_path)
    auth.register("alice", "a-long-unique-password", ["user"])
    token = auth.authenticate("alice", "a-long-unique-password")
    with pytest.raises(AuthorizationError):
        auth.authorize(token, ["admin"])
    auth.set_roles("alice", ["admin"])
    assert auth.authorize(token, ["admin"]).username == "alice"


@pytest.mark.parametrize("username", ["", "   ", None, 123])
def test_invalid_login_usernames_raise_authentication_error(tmp_path, username):
    auth = make_auth(tmp_path)
    with pytest.raises(AuthenticationError):
        auth.authenticate(username, "incorrect-password")


@pytest.mark.parametrize(
    "roles",
    ["admin", b"admin", [], [1], ["admin", []], None],
)
def test_invalid_role_collections_are_rejected(tmp_path, roles):
    auth = make_auth(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        auth.register("invalid-role-user", "a-long-unique-password", roles)


def test_role_names_are_deduplicated_and_sorted(tmp_path):
    auth = make_auth(tmp_path)
    auth.register(
        "role-user",
        "a-long-unique-password",
        ["user", "admin", "user"],
    )
    token = auth.authenticate("role-user", "a-long-unique-password")
    assert auth.authorize(token, ["admin", "user"]).username == "role-user"


def test_disabled_accounts_invalidate_existing_tokens(tmp_path):
    auth = make_auth(tmp_path)
    auth.register("alice", "a-long-unique-password")
    token = auth.authenticate("alice", "a-long-unique-password")
    auth.set_disabled("alice")
    with pytest.raises(AuthenticationError):
        auth.verify(token)


def test_token_tampering_is_rejected(tmp_path):
    auth = make_auth(tmp_path)
    auth.register("alice", "a-long-unique-password")
    token = auth.authenticate("alice", "a-long-unique-password")
    changed = ("A" if token[0] != "A" else "B") + token[1:]
    with pytest.raises(AuthenticationError):
        auth.verify(changed)


def test_token_revocation_survives_service_restart(tmp_path):
    auth = make_auth(tmp_path)
    auth.register("alice", "a-long-unique-password")
    token = auth.authenticate("alice", "a-long-unique-password")
    auth.revoke(token)
    restarted = make_auth(tmp_path)
    with pytest.raises(AuthenticationError):
        restarted.verify(token)


def test_token_expiry_is_enforced(tmp_path):
    now = [1000.0]
    auth = make_auth(tmp_path, clock=lambda: now[0])
    auth.register("alice", "a-long-unique-password")
    token = auth.authenticate("alice", "a-long-unique-password", ttl_seconds=10)
    assert auth.verify(token).username == "alice"
    now[0] += 11
    with pytest.raises(AuthenticationError):
        auth.verify(token)


def test_ledger_survives_restart_and_detects_tampering(tmp_path):
    path = tmp_path / "governance.jsonl"
    ledger = DurableGovernanceLedger(path, KEY)
    ledger.append("system", "test_event", {"safe": True})
    assert DurableGovernanceLedger(path, KEY).verify()
    records = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(records[0])
    record["details"]["safe"] = False
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        DurableGovernanceLedger(path, KEY)


def test_ledger_rejects_wrong_integrity_key(tmp_path):
    path = tmp_path / "governance.jsonl"
    DurableGovernanceLedger(path, KEY).append("system", "event")
    with pytest.raises(LedgerIntegrityError):
        DurableGovernanceLedger(path, b"different-integrity-key-32-bytes!!")


def test_recovery_requires_two_distinct_approvers(tmp_path):
    ledger = DurableGovernanceLedger(tmp_path / "audit.jsonl", KEY)
    recovery = GovernedRecovery(ledger)
    approvals = [
        ("alice", ["recovery_approver"]),
        ("ALICE", ["recovery_approver"]),
    ]
    with pytest.raises(PermissionError):
        recovery.run("proposal-1", approvals, lambda: True, lambda: True, lambda: True)
    assert "recovery_rejected" in (tmp_path / "audit.jsonl").read_text()


def test_recovery_verifies_and_records_success(tmp_path):
    ledger = DurableGovernanceLedger(tmp_path / "audit.jsonl", KEY)
    recovery = GovernedRecovery(ledger)
    approvals = [
        ("alice", ["recovery_approver"]),
        ("bob", ["recovery_approver"]),
    ]
    assert (
        recovery.run(
            "proposal-2", approvals, lambda: "recovered", lambda: True, lambda: True
        )
        == "recovered"
    )
    assert "recovery_succeeded" in (tmp_path / "audit.jsonl").read_text()


def test_failed_recovery_is_rolled_back_and_audited(tmp_path):
    ledger = DurableGovernanceLedger(tmp_path / "audit.jsonl", KEY)
    recovery = GovernedRecovery(ledger)
    approvals = [
        ("alice", ["recovery_approver"]),
        ("bob", ["recovery_approver"]),
    ]
    state = {"changed": False}

    def action():
        state["changed"] = True

    def rollback():
        state["changed"] = False
        return True

    with pytest.raises(RecoveryVerificationError):
        recovery.run("proposal-3", approvals, action, lambda: False, rollback)
    assert state["changed"] is False
    assert "recovery_rolled_back" in (tmp_path / "audit.jsonl").read_text()


def test_recovery_refuses_unverified_rollback(tmp_path):
    ledger = DurableGovernanceLedger(tmp_path / "audit.jsonl", KEY)
    recovery = GovernedRecovery(ledger)
    approvals = [
        ("alice", ["recovery_approver"]),
        ("bob", ["recovery_approver"]),
    ]
    with pytest.raises(RuntimeError, match="rollback was not verified"):
        recovery.run(
            "proposal-4", approvals, lambda: None, lambda: False, lambda: False
        )

def test_database_connections_are_closed(tmp_path, monkeypatch):
    import sqlite3

    from aviy.security import authentication as authentication_module

    connections = []
    original_connect = authentication_module.sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        authentication_module.sqlite3,
        "connect",
        tracked_connect,
    )

    auth = make_auth(tmp_path)
    auth.register("connection-user", "a-long-unique-password")
    token = auth.authenticate("connection-user", "a-long-unique-password")
    auth.verify(token)
    auth.set_roles("connection-user", ["admin"])
    auth.revoke(token)
    auth.set_disabled("connection-user")

    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
