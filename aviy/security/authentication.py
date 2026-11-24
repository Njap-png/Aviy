"""Persistent password authentication and signed access tokens."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class AuthenticationError(Exception):
    """Raised when credentials or an access token are invalid."""


class AuthorizationError(Exception):
    """Raised when an authenticated identity lacks permission."""


@dataclass(frozen=True)
class AuthenticatedIdentity:
    username: str
    roles: tuple[str, ...]
    token_id: str
    expires_at: int


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class AuthenticationService:
    """SQLite-backed accounts with PBKDF2 passwords and HMAC-signed tokens.

    Supply a persistent random signing key from a protected secret store.
    Never reuse the development key across deployments.
    """

    ITERATIONS = 310_000
    MAX_TOKEN_TTL = 86_400

    def __init__(
        self,
        database: str | Path,
        signing_key: bytes,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(signing_key, bytes) or len(signing_key) < 32:
            raise ValueError("signing_key must contain at least 32 bytes")
        self.database = Path(database).expanduser().absolute()
        if self.database.exists() and self.database.is_symlink():
            raise ValueError("Authentication database cannot be a symlink")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._key = bytes(signing_key)
        self._clock = clock
        self._dummy_salt = bytes(16)
        self._dummy_hash = hashlib.pbkdf2_hmac(
            "sha256", b"not-a-real-password", self._dummy_salt, self.ITERATIONS
        )
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a database connection and always close it after its transaction."""
        connection = sqlite3.connect(str(self.database), timeout=10)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 10000")
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS aviy_users (
                    username TEXT PRIMARY KEY,
                    salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    iterations INTEGER NOT NULL,
                    roles TEXT NOT NULL,
                    disabled INTEGER NOT NULL DEFAULT 0
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS aviy_revoked_tokens (
                    token_id TEXT PRIMARY KEY,
                    expires_at INTEGER NOT NULL
                )
            """)
        try:
            os.chmod(self.database, 0o600)
        except OSError:
            # Windows ACLs must be configured by the deployment administrator.
            pass

    @staticmethod
    def _username(value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Username must be text")
        value = value.strip().casefold()
        if not value or len(value) > 128 or any(ord(c) < 32 for c in value):
            raise ValueError("Invalid username")
        return value

    @staticmethod
    def _roles(values: Iterable[str]) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise TypeError("Roles must be an iterable of role names")

        try:
            supplied_roles = tuple(values)
        except TypeError as exc:
            raise ValueError("Roles must be an iterable of role names") from exc

        if not supplied_roles or any(
            not isinstance(role, str)
            or not role
            or len(role) > 64
            or not all(c.isalnum() or c in "_-." for c in role)
            for role in supplied_roles
        ):
            raise ValueError("At least one valid role is required")

        return tuple(sorted(set(supplied_roles)))

    def register(
        self, username: str, password: str, roles: Iterable[str] = ("user",)
    ) -> None:
        username = self._username(username)
        if not isinstance(password, str) or len(password) < 12:
            raise ValueError("Password must contain at least 12 characters")
        role_values = self._roles(roles)
        salt = secrets.token_bytes(16)
        password_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, self.ITERATIONS
        )
        try:
            with self._connect() as db:
                db.execute(
                    """INSERT INTO aviy_users
                       (username, salt, password_hash, iterations, roles)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        username,
                        salt,
                        password_hash,
                        self.ITERATIONS,
                        json.dumps(role_values),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Account already exists") from exc

    def _get_user(self, username: str):
        with self._connect() as db:
            return db.execute(
                "SELECT * FROM aviy_users WHERE username = ?", (username,)
            ).fetchone()

    def authenticate(
        self, username: str, password: str, *, ttl_seconds: int = 900
    ) -> str:
        try:
            username = self._username(username)
        except (TypeError, ValueError) as exc:
            raise AuthenticationError("Invalid username or password") from exc

        if not isinstance(password, str):
            raise AuthenticationError("Invalid username or password")
        if (
            not isinstance(ttl_seconds, int)
            or not 1 <= ttl_seconds <= self.MAX_TOKEN_TTL
        ):
            raise ValueError("Token lifetime must be between 1 and 86400 seconds")

        user = self._get_user(username)
        salt = user["salt"] if user else self._dummy_salt
        iterations = user["iterations"] if user else self.ITERATIONS
        expected = user["password_hash"] if user else self._dummy_hash
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        valid = hmac.compare_digest(actual, expected)
        if user is None or not valid or user["disabled"]:
            raise AuthenticationError("Invalid username or password")

        now = int(self._clock())
        payload = {
            "sub": username,
            "iat": now,
            "exp": now + ttl_seconds,
            "jti": secrets.token_urlsafe(24),
        }
        encoded = _b64encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        signature = _b64encode(
            hmac.new(self._key, encoded.encode("ascii"), hashlib.sha256).digest()
        )
        return encoded + "." + signature

    def verify(self, token: str) -> AuthenticatedIdentity:
        try:
            if not isinstance(token, str) or len(token) > 8192:
                raise ValueError
            encoded, supplied_signature = token.split(".", 1)
            expected_signature = _b64encode(
                hmac.new(self._key, encoded.encode("ascii"), hashlib.sha256).digest()
            )
            if not hmac.compare_digest(supplied_signature, expected_signature):
                raise ValueError
            payload = json.loads(_b64decode(encoded))
            username = self._username(payload["sub"])
            token_id = payload["jti"]
            issued = payload["iat"]
            expires = payload["exp"]
            if (
                not isinstance(token_id, str)
                or not token_id
                or type(issued) is not int
                or type(expires) is not int
                or expires <= int(self._clock())
                or issued > int(self._clock()) + 60
                or expires <= issued
                or expires - issued > self.MAX_TOKEN_TTL
            ):
                raise ValueError
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, UnicodeError):
            raise AuthenticationError("Invalid or expired access token") from None

        user = self._get_user(username)
        if user is None or user["disabled"]:
            raise AuthenticationError("Invalid or expired access token")
        with self._connect() as db:
            revoked = db.execute(
                "SELECT 1 FROM aviy_revoked_tokens WHERE token_id = ?",
                (token_id,),
            ).fetchone()
        if revoked:
            raise AuthenticationError("Invalid or expired access token")

        return AuthenticatedIdentity(
            username=username,
            roles=tuple(json.loads(user["roles"])),
            token_id=token_id,
            expires_at=expires,
        )

    def authorize(
        self, token: str, required_roles: Iterable[str]
    ) -> AuthenticatedIdentity:
        required = set(self._roles(required_roles))
        identity = self.verify(token)
        if not required.intersection(identity.roles):
            raise AuthorizationError("Insufficient permissions")
        return identity

    def revoke(self, token: str) -> None:
        identity = self.verify(token)
        with self._connect() as db:
            db.execute(
                "INSERT INTO aviy_revoked_tokens (token_id, expires_at) VALUES (?, ?)",
                (identity.token_id, identity.expires_at),
            )

    def set_roles(self, username: str, roles: Iterable[str]) -> None:
        username = self._username(username)
        role_json = json.dumps(self._roles(roles))
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE aviy_users SET roles = ? WHERE username = ?",
                (role_json, username),
            )
            if cursor.rowcount != 1:
                raise KeyError("Account does not exist")

    def set_disabled(self, username: str, disabled: bool = True) -> None:
        username = self._username(username)
        if type(disabled) is not bool:
            raise ValueError("disabled must be a boolean")
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE aviy_users SET disabled = ? WHERE username = ?",
                (int(disabled), username),
            )
            if cursor.rowcount != 1:
                raise KeyError("Account does not exist")
