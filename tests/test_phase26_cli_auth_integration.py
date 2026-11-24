import pytest

from aviy.cli.main import _make_validator
from aviy.security.authentication import (
    AuthenticationError,
    AuthenticationService,
)


def test_cli_validator_uses_real_persistent_authentication(tmp_path, monkeypatch):
    database = tmp_path / "auth.sqlite3"
    key = "test-signing-key-with-more-than-32-bytes"
    monkeypatch.setenv("AVIY_TOKEN_SIGNING_KEY", key)
    monkeypatch.setenv("AVIY_AUTH_DB", str(database))

    service = AuthenticationService(
        database=database,
        signing_key=key.encode("utf-8"),
    )
    service.register("integration-user", "Strong-test-password-123", ["admin"])
    token = service.authenticate("integration-user", "Strong-test-password-123")

    validator = _make_validator()
    identity = validator(token)
    assert identity is not None

    with pytest.raises(AuthenticationError):
        validator("not-a-valid-token")

    service.revoke(token)
    with pytest.raises(AuthenticationError):
        validator(token)


def test_cli_validator_rejects_missing_or_short_signing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("AVIY_TOKEN_SIGNING_KEY", raising=False)
    monkeypatch.setenv("AVIY_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    with pytest.raises(RuntimeError):
        _make_validator()

    monkeypatch.setenv("AVIY_TOKEN_SIGNING_KEY", "too-short")
    with pytest.raises(RuntimeError):
        _make_validator()
