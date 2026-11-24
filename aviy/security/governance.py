"""Explicit authorization, secret resolution, network and filesystem boundaries."""

import copy
import hashlib
import hmac
import ipaddress
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Principal:
    subject: str
    roles: frozenset

    def __post_init__(self):
        if not isinstance(self.subject, str) or not self.subject.strip():
            raise ValueError("Principal subject must be a non-empty string.")
        if (
            not isinstance(self.roles, (set, frozenset, tuple, list))
            or not self.roles
            or any(not isinstance(role, str) or not role.strip() for role in self.roles)
        ):
            raise ValueError("Principal needs at least one valid role.")
        object.__setattr__(self, "roles", frozenset(self.roles))


class AuthorizationPolicy:
    def __init__(self, permissions):
        if not isinstance(permissions, dict):
            raise TypeError("permissions must be a dictionary.")
        normalized = {}
        for role, values in permissions.items():
            if not isinstance(role, str) or not role.strip():
                raise ValueError("Permission roles must be non-empty strings.")
            if (
                not isinstance(values, (set, frozenset, tuple, list))
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise ValueError("Each role must map to a collection of permissions.")
            normalized[role] = frozenset(values)
        self.permissions = normalized

    def authorize(self, principal, permission):
        if not isinstance(principal, Principal):
            raise TypeError("A Principal is required.")
        if not isinstance(permission, str) or not permission:
            raise ValueError("Permission is required.")
        return any(
            permission in self.permissions.get(role, ()) for role in principal.roles
        )

    def require(self, principal, permission):
        if not self.authorize(principal, permission):
            raise PermissionError("Access denied.")


class EnvironmentSecretResolver:
    """Resolves secret values at runtime; never writes or logs their contents."""

    def __init__(self, environ=None):
        self.environ = os.environ if environ is None else environ

    def resolve(self, name):
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", name, flags=re.ASCII
        ):
            raise ValueError("Invalid secret environment variable name.")
        value = self.environ.get(name)
        if not isinstance(value, str) or not value:
            raise KeyError(f"Secret is unavailable: {name}")
        return value


class NetworkPolicy:
    def __init__(self, allowed_hosts):
        try:
            hosts = tuple(allowed_hosts)
        except TypeError as exc:
            raise TypeError("allowed_hosts must be an iterable of host strings.") from exc
        if not hosts or any(not isinstance(host, str) or not host.strip() for host in hosts):
            raise ValueError("At least one valid approved host is required.")
        self.allowed_hosts = frozenset(host.lower().rstrip(".") for host in hosts)

    def validate(self, url):
        if not isinstance(url, str) or not url:
            raise PermissionError("A valid HTTPS URL is required.")
        try:
            parsed = urlsplit(url)
            host = parsed.hostname
            port = parsed.port
        except ValueError as exc:
            raise PermissionError("Malformed URL.") from exc
        if (
            parsed.scheme.lower() != "https"
            or not host
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise PermissionError("Only credential-free HTTPS URLs are allowed.")
        host = host.lower().rstrip(".")
        if host not in self.allowed_hosts:
            raise PermissionError("Host is not allowlisted.")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise PermissionError("Non-public network addresses are forbidden.")
        if port not in (None, 443):
            raise PermissionError("Non-standard HTTPS ports are forbidden.")
        return True


class FilesystemBoundary:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def validate(self, relative_path):
        if not isinstance(relative_path, (str, Path)) or not str(relative_path):
            raise PermissionError("A non-empty relative path is required.")
        windows_path = PureWindowsPath(str(relative_path))
        candidate = Path(relative_path)
        if (
            candidate.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or windows_path.root
            or not candidate.parts
            or ".." in candidate.parts
        ):
            raise PermissionError("Unsafe relative path.")
        current = self.root
        for part in candidate.parts:
            current = current / part
            if current.is_symlink():
                raise PermissionError("Symlinks are forbidden inside the boundary.")
        resolved = current.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise PermissionError("Path escapes the filesystem boundary.")
        return resolved


class GovernanceLedger:
    """Append-only in-memory hash-linked decision history."""

    def __init__(self):
        self._entries = []
        self._head = "0" * 64

    def append(self, actor, action, decision, evidence=None):
        if not all(isinstance(x, str) and x.strip() for x in (actor, action, decision)):
            raise ValueError("Ledger identity and decision fields are required.")
        if evidence is None:
            evidence = {}
        if not isinstance(evidence, dict):
            raise TypeError("evidence must be a dictionary.")
        try:
            safe_evidence = json.loads(json.dumps(
                evidence, sort_keys=True, separators=(",", ":"), allow_nan=False
            ))
        except (TypeError, ValueError) as exc:
            raise ValueError("evidence must contain finite JSON values.") from exc
        payload = {
            "sequence": len(self._entries) + 1,
            "actor": actor,
            "action": action,
            "decision": decision,
            "evidence": safe_evidence,
            "previous_hash": self._head,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        self._entries.append({"payload": payload, "hash": digest})
        self._head = digest
        return digest

    def verify(self):
        previous = "0" * 64
        for index, entry in enumerate(self._entries, 1):
            payload = entry["payload"]
            if payload["sequence"] != index or payload["previous_hash"] != previous:
                return False
            encoded = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            digest = hashlib.sha256(encoded.encode()).hexdigest()
            if not hmac.compare_digest(digest, entry["hash"]):
                return False
            previous = digest
        return hmac.compare_digest(previous, self._head)

    @property
    def entries(self):
        return tuple(copy.deepcopy(entry) for entry in self._entries)


class DualApproval:
    REQUIRED_ROLE = "approver"

    def approve(self, approvers):
        if not isinstance(approvers, (tuple, list, set, frozenset)):
            raise TypeError("approvers must be a collection of Principal objects.")
        if any(not isinstance(person, Principal) for person in approvers):
            raise TypeError("Every approver must be a Principal.")
        qualified = {
            person.subject.casefold()
            for person in approvers
            if self.REQUIRED_ROLE in person.roles
        }
        if len(qualified) < 2:
            raise PermissionError("Two distinct authorized approvers are required.")
        return True
