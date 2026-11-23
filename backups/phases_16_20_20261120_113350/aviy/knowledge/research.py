from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Protocol
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ResearchPlan:
    question: str
    steps: tuple[str, ...]
    uncertainty: str


@dataclass(frozen=True)
class ResearchSource:
    source_id: str
    uri: str
    title: str
    license: str
    content: str
    content_sha256: str
    verified: bool
    uncertainty: str


class SourceDiscovery(Protocol):
    def discover(self, query: str) -> list[ResearchSource]: ...


def validate_source_uri(uri: str) -> bool:
    try:
        parsed = urlsplit(uri)
    except ValueError:
        return False
    return (
        parsed.scheme in ("https", "http")
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
    )


class ResourceAcquirer:
    def __init__(
        self,
        *,
        allowed_licenses: set[str],
        allowed_hosts: set[str],
        maximum_bytes: int = 1_000_000,
    ) -> None:
        if maximum_bytes < 1:
            raise ValueError("maximum_bytes must be positive.")
        self.allowed_licenses = {item.casefold() for item in allowed_licenses}
        self.allowed_hosts = {item.casefold() for item in allowed_hosts}
        self.maximum_bytes = maximum_bytes

    def acquire(
        self,
        *,
        source_id: str,
        uri: str,
        title: str,
        license: str,
        content: bytes,
    ) -> ResearchSource:
        if not validate_source_uri(uri):
            raise ValueError("Source URI is invalid or unsafe.")
        host = (urlsplit(uri).hostname or "").casefold()
        if host not in self.allowed_hosts:
            raise PermissionError("Source host is not approved.")
        if license.casefold() not in self.allowed_licenses:
            raise PermissionError("Source license is not approved.")
        if len(content) > self.maximum_bytes:
            raise ValueError("Resource exceeds the acquisition size limit.")
        text = content.decode("utf-8", errors="strict")
        digest = hashlib.sha256(content).hexdigest()
        return ResearchSource(
            source_id=source_id,
            uri=uri,
            title=title,
            license=license,
            content=text,
            content_sha256=digest,
            verified=False,
            uncertainty="Content integrity is known; factual claims remain unverified.",
        )


class ResearchEngine:
    def __init__(
        self,
        discover: Callable[[str], list[ResearchSource]],
        verify: Callable[[ResearchSource], bool],
    ) -> None:
        self._discover = discover
        self._verify = verify

    def plan(self, question: str) -> ResearchPlan:
        if not question.strip():
            raise ValueError("Research question is required.")
        return ResearchPlan(
            question=question,
            steps=(
                "Discover candidate sources",
                "Check source provenance and licensing",
                "Verify claims against available evidence",
                "Record citations and unresolved uncertainty",
            ),
            uncertainty="Unresolved until evidence is independently evaluated.",
        )

    def run(self, question: str) -> tuple[ResearchSource, ...]:
        self.plan(question)
        sources = self._discover(question)
        results = []
        seen: set[str] = set()
        for source in sources:
            if source.source_id in seen:
                continue
            seen.add(source.source_id)
            verified = bool(self._verify(source))
            results.append(
                ResearchSource(
                    source_id=source.source_id,
                    uri=source.uri,
                    title=source.title,
                    license=source.license,
                    content=source.content,
                    content_sha256=source.content_sha256,
                    verified=verified,
                    uncertainty=(
                        "Verifier accepted this source; claims still require "
                        "appropriate corroboration."
                        if verified else
                        "Source was not verified; do not treat its claims as facts."
                    ),
                )
            )
        return tuple(results)
