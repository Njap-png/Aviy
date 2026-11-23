from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ResearchPlan:
    question: str
    steps: tuple[str, ...]
    uncertainty: str


@dataclass(frozen=True)
class Citation:
    citation_id: str
    uri: str
    title: str
    publisher: str = ""
    locator: str = ""


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
    publisher: str = ""
    retrieved_at: str = ""
    citations: tuple[Citation, ...] = ()


class SourceDiscovery(Protocol):
    def discover(self, query: str) -> list[ResearchSource]: ...


def validate_source_uri(uri: str) -> bool:
    try:
        parsed = urlsplit(uri)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme in ("https", "http")
        and bool(hostname)
        and not parsed.username
        and not parsed.password
        and not parsed.fragment
    )


class ResourceAcquirer:
    def __init__(
        self,
        *,
        allowed_licenses: set[str],
        allowed_hosts: set[str],
        maximum_bytes: int = 1_000_000,
    ) -> None:
        if (
            isinstance(maximum_bytes, bool)
            or not isinstance(maximum_bytes, int)
            or maximum_bytes < 1
        ):
            raise ValueError("maximum_bytes must be a positive integer.")
        if not isinstance(allowed_licenses, set) or not isinstance(allowed_hosts, set):
            raise TypeError("Allowed licenses and hosts must be sets.")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in (*allowed_licenses, *allowed_hosts)
        ):
            raise ValueError("Allowed licenses and hosts must be non-empty strings.")
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
        publisher: str = "",
        citations: tuple[Citation, ...] = (),
        retrieved_at: str | None = None,
    ) -> ResearchSource:
        if (
            not isinstance(source_id, str) or not source_id.strip()
            or not isinstance(title, str) or not title.strip()
        ):
            raise ValueError("Source identity and title are required strings.")
        if not isinstance(uri, str) or not validate_source_uri(uri):
            raise ValueError("Source URI is invalid or unsafe.")
        if not isinstance(license, str) or not license.strip():
            raise ValueError("A non-empty license identifier is required.")
        if not isinstance(publisher, str):
            raise TypeError("Publisher must be a string.")
        if not isinstance(content, bytes):
            raise TypeError("Resource content must be bytes.")
        if not isinstance(citations, tuple):
            raise TypeError("Citations must be supplied as a tuple.")
        host = (urlsplit(uri).hostname or "").casefold()
        if host not in self.allowed_hosts:
            raise PermissionError("Source host is not approved.")
        if license.casefold() not in self.allowed_licenses:
            raise PermissionError("Source license is not approved.")
        if len(content) > self.maximum_bytes:
            raise ValueError("Resource exceeds the acquisition size limit.")
        for citation in citations:
            if (
                not isinstance(citation, Citation)
                or not isinstance(citation.citation_id, str)
                or not citation.citation_id.strip()
                or not isinstance(citation.title, str)
                or not citation.title.strip()
                or not isinstance(citation.uri, str)
                or not validate_source_uri(citation.uri)
                or not isinstance(citation.publisher, str)
                or not isinstance(citation.locator, str)
            ):
                raise ValueError("Citation metadata is invalid.")
        text = content.decode("utf-8", errors="strict")
        digest = hashlib.sha256(content).hexdigest()
        timestamp = retrieved_at or datetime.now(UTC).isoformat()
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError("Retrieved timestamp must be valid ISO 8601.") from exc
        if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
            raise ValueError("Retrieved timestamp must include a timezone.")
        timestamp = parsed_timestamp.isoformat()
        return ResearchSource(
            source_id=source_id,
            uri=uri,
            title=title,
            license=license,
            content=text,
            content_sha256=digest,
            verified=False,
            uncertainty="Content integrity is known; factual claims remain unverified.",
            publisher=publisher,
            retrieved_at=timestamp,
            citations=tuple(citations),
        )


class ResearchEngine:
    def __init__(
        self,
        discover: Callable[[str], list[ResearchSource]],
        verify: Callable[[ResearchSource], bool],
        *,
        allowed_licenses: set[str] | None = None,
    ) -> None:
        if not callable(discover) or not callable(verify):
            raise TypeError("Discovery and verification providers must be callable.")
        if allowed_licenses is not None and (
            not isinstance(allowed_licenses, set)
            or any(
                not isinstance(item, str) or not item.strip()
                for item in allowed_licenses
            )
        ):
            raise ValueError("Allowed licenses must be non-empty strings.")
        self._discover = discover
        self._verify = verify
        self._allowed_licenses = (
            {item.casefold() for item in allowed_licenses}
            if allowed_licenses is not None else None
        )

    def plan(self, question: str) -> ResearchPlan:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Research question must be a non-empty string.")
        return ResearchPlan(
            question=question,
            steps=(
                "Discover candidate sources",
                "Validate source identity, URI and licensing",
                "Verify content integrity and source provenance",
                "Evaluate evidence and preserve citations",
                "Record unresolved uncertainty",
            ),
            uncertainty="Unresolved until evidence is independently evaluated.",
        )

    def run(self, question: str) -> tuple[ResearchSource, ...]:
        self.plan(question)
        sources = self._discover(question)
        if not isinstance(sources, (list, tuple)):
            raise TypeError("Source discovery must return a list or tuple.")
        results: list[ResearchSource] = []
        seen: dict[str, ResearchSource] = {}
        for source in sources:
            if not isinstance(source, ResearchSource):
                raise TypeError("Discovery returned an invalid source record.")
            if (
                not isinstance(source.source_id, str)
                or not source.source_id.strip()
                or not isinstance(source.title, str)
                or not source.title.strip()
                or not isinstance(source.uri, str)
                or not validate_source_uri(source.uri)
                or not isinstance(source.license, str)
                or not source.license.strip()
                or not isinstance(source.content, str)
                or not isinstance(source.content_sha256, str)
                or not isinstance(source.uncertainty, str)
                or not isinstance(source.publisher, str)
                or not isinstance(source.retrieved_at, str)
                or not isinstance(source.citations, tuple)
                or not isinstance(source.verified, bool)
            ):
                raise ValueError("Discovery returned incomplete or invalid source metadata.")
            if source.source_id in seen:
                if source != seen[source.source_id]:
                    raise ValueError(
                        "Discovery returned conflicting records for one source ID."
                    )
                continue
            seen[source.source_id] = source
            if not source.source_id.strip() or not validate_source_uri(source.uri):
                raise ValueError("Discovery returned an invalid source.")
            if self._allowed_licenses is not None and (
                source.license.casefold() not in self._allowed_licenses
            ):
                raise PermissionError("Discovered source has an unapproved license.")
            if (
                len(source.content_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in source.content_sha256.casefold()
                )
            ):
                raise ValueError("Discovered source content hash is malformed.")
            actual_hash = hashlib.sha256(source.content.encode("utf-8")).hexdigest()
            if actual_hash != source.content_sha256.casefold():
                raise ValueError("Discovered source content hash does not match.")
            if any(
                not isinstance(item, Citation)
                or not isinstance(item.citation_id, str)
                or not item.citation_id.strip()
                or not isinstance(item.title, str)
                or not item.title.strip()
                or not isinstance(item.uri, str)
                or not validate_source_uri(item.uri)
                or not isinstance(item.publisher, str)
                or not isinstance(item.locator, str)
                for item in source.citations
            ):
                raise ValueError("Discovered source contains invalid citation metadata.")
            verified = self._verify(source)
            if not isinstance(verified, bool):
                raise TypeError("Source verifier must return a Boolean.")
            results.append(ResearchSource(
                source_id=source.source_id,
                uri=source.uri,
                title=source.title,
                license=source.license,
                content=source.content,
                content_sha256=actual_hash,
                verified=verified,
                uncertainty=(
                    (
                        source.uncertainty.strip()
                        if isinstance(source.uncertainty, str)
                        and source.uncertainty.strip()
                        else "No source-specific uncertainty was supplied."
                    )
                    + (
                        " Source verification passed; individual claims still "
                        "require appropriate corroboration."
                        if verified else
                        " Source was not verified; do not treat its claims as facts."
                    )
                ),
                publisher=source.publisher,
                retrieved_at=source.retrieved_at,
                citations=source.citations,
            ))
        return tuple(results)
