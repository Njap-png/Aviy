from __future__ import annotations

import hashlib
import math
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
            not isinstance(source_id, str)
            or not source_id.strip()
            or not isinstance(title, str)
            or not title.strip()
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
            if allowed_licenses is not None
            else None
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
                raise ValueError(
                    "Discovery returned incomplete or invalid source metadata."
                )
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
            if len(source.content_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in source.content_sha256.casefold()
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
                raise ValueError(
                    "Discovered source contains invalid citation metadata."
                )
            verified = self._verify(source)
            if not isinstance(verified, bool):
                raise TypeError("Source verifier must return a Boolean.")
            results.append(
                ResearchSource(
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
                            if verified
                            else " Source was not verified; do not treat its claims as facts."
                        )
                    ),
                    publisher=source.publisher,
                    retrieved_at=source.retrieved_at,
                    citations=source.citations,
                )
            )
        return tuple(results)


# ---------------------------------------------------------------------------
# Controlled HTTPS source discovery and acquisition
# ---------------------------------------------------------------------------

import http.client
import ipaddress
import socket
import ssl
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDescriptor:
    """An explicitly approved research source in the discovery catalogue."""

    source_id: str
    uri: str
    title: str
    license: str
    publisher: str
    citations: tuple[Citation, ...] = ()


def _public_addresses(host: str, port: int) -> tuple[str, ...]:
    """Resolve a hostname and reject non-public or unsafe destinations."""
    try:
        records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise PermissionError("Source hostname could not be resolved.") from exc

    resolved_addresses: list[str] = []
    for record in records:
        candidate = record[4][0]
        if isinstance(candidate, str):
            resolved_addresses.append(candidate)

    addresses = tuple(dict.fromkeys(resolved_addresses))
    if not addresses:
        raise PermissionError("Source hostname has no usable address.")

    for address in addresses:
        try:
            parsed = ipaddress.ip_address(address.split("%", 1)[0])
        except ValueError as exc:
            raise PermissionError("Source resolved to an invalid IP address.") from exc
        if not parsed.is_global:
            raise PermissionError(
                "Private, loopback, link-local and reserved destinations are forbidden."
            )
    return addresses


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection pinned to a previously validated public IP."""

    def __init__(self, hostname: str, address: str, port: int, timeout: float):
        self._ssl_context = ssl.create_default_context()
        super().__init__(
            hostname,
            port=port,
            timeout=timeout,
            context=self._ssl_context,
        )
        self._pinned_address = address

    def connect(self):
        raw_socket = socket.create_connection(
            (self._pinned_address, self.port),
            self.timeout,
        )
        try:
            self.sock = self._ssl_context.wrap_socket(
                raw_socket,
                server_hostname=self.host,
            )
        except BaseException:
            raw_socket.close()
            raise


class HTTPSResourceFetcher:
    """Fetch bounded HTTPS resources without following redirects or proxies."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        maximum_bytes: int = 1_000_000,
    ):
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive.")
        if (
            isinstance(maximum_bytes, bool)
            or not isinstance(maximum_bytes, int)
            or maximum_bytes < 1
        ):
            raise ValueError("maximum_bytes must be a positive integer.")
        self.timeout_seconds = float(timeout_seconds)
        self.maximum_bytes = maximum_bytes

    def fetch(self, uri: str) -> bytes:
        if not isinstance(uri, str) or not validate_source_uri(uri):
            raise ValueError("Source URI is invalid.")
        parsed = urlsplit(uri)
        if parsed.scheme.casefold() != "https":
            raise PermissionError("Only HTTPS source acquisition is permitted.")
        if parsed.query and len(parsed.query) > 4096:
            raise ValueError("Source query string is too large.")
        if parsed.port not in (None, 443):
            raise PermissionError("Only the standard HTTPS port is permitted.")

        host = parsed.hostname
        if not host:
            raise ValueError("Source hostname is required.")

        addresses = _public_addresses(host, 443)
        # Reject mixed public/private DNS answers instead of choosing one.
        connection = _PinnedHTTPSConnection(
            host, addresses[0], 443, self.timeout_seconds
        )
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query

        try:
            connection.request(
                "GET",
                path,
                headers={
                    "Host": host,
                    "User-Agent": "Aviy-Research/1.0",
                    "Accept": "text/plain, text/html, application/json, application/xml",
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            if response.status != 200:
                # Redirects are deliberately rejected; callers must approve
                # the final destination explicitly.
                raise PermissionError(
                    f"Source server returned HTTP status {response.status}."
                )
            content_type = response.getheader("Content-Type", "")
            media_type = content_type.split(";", 1)[0].strip().casefold()
            if media_type not in {
                "text/plain",
                "text/html",
                "application/json",
                "application/xml",
                "text/xml",
            }:
                raise ValueError("Source returned an unsupported content type.")

            length = response.getheader("Content-Length")
            if length is not None:
                try:
                    if int(length) > self.maximum_bytes:
                        raise ValueError("Source exceeds the acquisition size limit.")
                except ValueError as exc:
                    if "size limit" in str(exc):
                        raise
                    raise ValueError("Source has an invalid Content-Length.") from exc

            content = response.read(self.maximum_bytes + 1)
            if len(content) > self.maximum_bytes:
                raise ValueError("Source exceeds the acquisition size limit.")
            return content
        finally:
            connection.close()


class HTTPSourceDiscovery:
    """
    Discover sources from an explicitly approved catalogue and acquire them.

    The optional fetcher argument is intended for controlled tests and
    alternative transports. Production use defaults to HTTPSResourceFetcher.
    """

    def __init__(
        self,
        catalogue: tuple[SourceDescriptor, ...] | list[SourceDescriptor],
        *,
        allowed_licenses: set[str],
        allowed_hosts: set[str],
        maximum_bytes: int = 1_000_000,
        timeout_seconds: float = 10.0,
        fetcher: Callable[[str], bytes] | None = None,
    ):
        if not isinstance(catalogue, (tuple, list)):
            raise TypeError("Catalogue must be a list or tuple.")
        if not isinstance(allowed_licenses, set) or not isinstance(allowed_hosts, set):
            raise TypeError("Allowed licenses and hosts must be sets.")
        self.allowed_licenses = {
            item.casefold()
            for item in allowed_licenses
            if isinstance(item, str) and item.strip()
        }
        self.allowed_hosts = {
            item.casefold()
            for item in allowed_hosts
            if isinstance(item, str) and item.strip()
        }
        if len(self.allowed_licenses) != len(allowed_licenses):
            raise ValueError("Allowed licenses must contain non-empty strings.")
        if len(self.allowed_hosts) != len(allowed_hosts):
            raise ValueError("Allowed hosts must contain non-empty strings.")

        self.acquirer = ResourceAcquirer(
            allowed_licenses=set(allowed_licenses),
            allowed_hosts=set(allowed_hosts),
            maximum_bytes=maximum_bytes,
        )
        self.catalogue = tuple(catalogue)
        self.fetcher = (
            fetcher
            or HTTPSResourceFetcher(
                timeout_seconds=timeout_seconds,
                maximum_bytes=maximum_bytes,
            ).fetch
        )

        identities = set()
        for item in self.catalogue:
            if not isinstance(item, SourceDescriptor):
                raise TypeError("Catalogue contains an invalid source descriptor.")
            if (
                not item.source_id.strip()
                or not item.title.strip()
                or not item.publisher.strip()
                or not item.license.strip()
                or not validate_source_uri(item.uri)
            ):
                raise ValueError("Catalogue source metadata is incomplete.")
            parsed = urlsplit(item.uri)
            if parsed.scheme.casefold() != "https":
                raise PermissionError("Catalogue sources must use HTTPS.")
            if (parsed.hostname or "").casefold() not in self.allowed_hosts:
                raise PermissionError("Catalogue source host is not approved.")
            if item.license.casefold() not in self.allowed_licenses:
                raise PermissionError("Catalogue source license is not approved.")
            if item.source_id in identities:
                raise ValueError("Catalogue contains duplicate source identities.")
            identities.add(item.source_id)

    def discover(self, query: str) -> list[ResearchSource]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Research query must be a non-empty string.")

        terms = tuple(term.casefold() for term in query.split() if term.strip())
        results = []
        for descriptor in self.catalogue:
            searchable = (
                descriptor.title
                + " "
                + descriptor.publisher
                + " "
                + descriptor.source_id
            ).casefold()
            if not all(term in searchable for term in terms):
                continue

            content = self.fetcher(descriptor.uri)
            if not isinstance(content, bytes):
                raise TypeError("Resource fetcher must return bytes.")

            results.append(
                self.acquirer.acquire(
                    source_id=descriptor.source_id,
                    uri=descriptor.uri,
                    title=descriptor.title,
                    license=descriptor.license,
                    content=content,
                    publisher=descriptor.publisher,
                    citations=descriptor.citations,
                )
            )
        return results


def build_research_engine(
    catalogue: tuple[SourceDescriptor, ...] | list[SourceDescriptor],
    *,
    allowed_licenses: set[str],
    allowed_hosts: set[str],
    verifier: Callable[[ResearchSource], bool],
    maximum_bytes: int = 1_000_000,
    timeout_seconds: float = 10.0,
    fetcher: Callable[[str], bytes] | None = None,
) -> ResearchEngine:
    """Construct an end-to-end engine with controlled source discovery."""
    discovery = HTTPSourceDiscovery(
        catalogue,
        allowed_licenses=allowed_licenses,
        allowed_hosts=allowed_hosts,
        maximum_bytes=maximum_bytes,
        timeout_seconds=timeout_seconds,
        fetcher=fetcher,
    )
    return ResearchEngine(
        discover=discovery.discover,
        verify=verifier,
        allowed_licenses=allowed_licenses,
    )
