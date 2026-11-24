"""End-to-end tests for controlled Phase 18 research acquisition."""



import hashlib
import socket
from typing import Any, cast

import pytest

from aviy.knowledge.research import (
    Citation,
    HTTPSourceDiscovery,
    HTTPSResourceFetcher,
    SourceDescriptor,
    _public_addresses,
    build_research_engine,
)


def make_catalogue():

    return (

        SourceDescriptor(

            source_id="paper-001",

            uri="https://research.example.org/paper",

            title="Aviy Mathematics Research",

            license="MIT",

            publisher="Example Research",

            citations=(

                Citation(

                    "citation-001",

                    "https://research.example.org/paper",

                    "Primary research paper",

                    "Example Research",

                    "section 2",

                ),

            ),

        ),

    )





def test_end_to_end_research_preserves_provenance_and_uncertainty():

    payload = b"Finding: the measured result is 42."

    calls = []



    def fetch(uri):

        calls.append(uri)

        return payload



    def verify(source):

        assert source.content_sha256 == hashlib.sha256(payload).hexdigest()

        return True



    engine = build_research_engine(

        make_catalogue(),

        allowed_licenses={"MIT"},

        allowed_hosts={"research.example.org"},

        verifier=verify,

        fetcher=fetch,

    )



    plan = engine.plan("Aviy Mathematics Research")

    results = engine.run("Aviy Mathematics Research")



    assert plan.question == "Aviy Mathematics Research"

    assert len(calls) == 1

    assert len(results) == 1



    source = results[0]

    assert source.source_id == "paper-001"

    assert source.publisher == "Example Research"

    assert source.license == "MIT"

    assert source.content == payload.decode("utf-8")

    assert source.content_sha256 == hashlib.sha256(payload).hexdigest()

    assert source.retrieved_at

    assert source.verified is True

    assert len(source.citations) == 1

    assert source.citations[0].citation_id == "citation-001"

    assert "corroboration" in source.uncertainty





def test_discovery_does_not_fetch_unmatched_catalogue_sources():

    calls = []

    discovery = HTTPSourceDiscovery(

        make_catalogue(),

        allowed_licenses={"MIT"},

        allowed_hosts={"research.example.org"},

        fetcher=lambda uri: calls.append(uri) or b"content",

    )



    assert discovery.discover("no matching source") == []

    assert calls == []





def test_discovery_rejects_unapproved_catalogue_host():

    descriptor = SourceDescriptor(

        "bad", "https://evil.example/data", "Research", "MIT", "Publisher"

    )

    with pytest.raises(PermissionError, match="host"):

        HTTPSourceDiscovery(

            (descriptor,),

            allowed_licenses={"MIT"},

            allowed_hosts={"research.example.org"},

            fetcher=lambda uri: b"content",

        )





def test_discovery_rejects_unapproved_license():

    descriptor = SourceDescriptor(

        "bad", "https://research.example.org/data", "Research", "UNKNOWN", "Publisher"

    )

    with pytest.raises(PermissionError, match="license"):

        HTTPSourceDiscovery(

            (descriptor,),

            allowed_licenses={"MIT"},

            allowed_hosts={"research.example.org"},

            fetcher=lambda uri: b"content",

        )





def test_discovery_rejects_non_https_catalogue_uri():

    descriptor = SourceDescriptor(

        "bad", "http://research.example.org/data", "Research", "MIT", "Publisher"

    )

    with pytest.raises(PermissionError, match="HTTPS"):

        HTTPSourceDiscovery(

            (descriptor,),

            allowed_licenses={"MIT"},

            allowed_hosts={"research.example.org"},

            fetcher=lambda uri: b"content",

        )





def test_discovery_rejects_non_bytes_fetcher_result():

    discovery = HTTPSourceDiscovery(

        make_catalogue(),

        allowed_licenses={"MIT"},

        allowed_hosts={"research.example.org"},

        fetcher=cast(Any, lambda uri: "not bytes"),

    )

    with pytest.raises(TypeError, match="bytes"):

        discovery.discover("Aviy Mathematics Research")





def test_discovery_enforces_resource_size_limit():

    discovery = HTTPSourceDiscovery(

        make_catalogue(),

        allowed_licenses={"MIT"},

        allowed_hosts={"research.example.org"},

        maximum_bytes=3,

        fetcher=lambda uri: b"too large",

    )

    with pytest.raises(ValueError, match="size limit"):

        discovery.discover("Aviy Mathematics Research")





def test_https_fetcher_rejects_http():

    fetcher = HTTPSResourceFetcher()

    with pytest.raises(PermissionError, match="HTTPS"):

        fetcher.fetch("http://research.example.org/data")





def test_https_fetcher_rejects_invalid_uri():

    fetcher = HTTPSResourceFetcher()

    with pytest.raises(ValueError, match="URI"):

        fetcher.fetch("file:///etc/passwd")





def test_public_address_validation_rejects_private_dns(monkeypatch):

    monkeypatch.setattr(

        socket,

        "getaddrinfo",

        lambda *args, **kwargs: [

            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))

        ],

    )

    with pytest.raises(PermissionError, match="Private"):

        _public_addresses("localhost", 443)





def test_research_engine_does_not_claim_unverified_source_is_fact():

    payload = b"uncertain finding"

    engine = build_research_engine(

        make_catalogue(),

        allowed_licenses={"MIT"},

        allowed_hosts={"research.example.org"},

        verifier=lambda source: False,

        fetcher=lambda uri: payload,

    )

    source = engine.run("Aviy Mathematics Research")[0]

    assert source.verified is False

    assert "do not treat its claims as facts" in source.uncertainty





def test_research_engine_preserves_license_and_citation():

    engine = build_research_engine(

        make_catalogue(),

        allowed_licenses={"MIT"},

        allowed_hosts={"research.example.org"},

        verifier=lambda source: True,

        fetcher=lambda uri: b"finding",

    )

    source = engine.run("Aviy Mathematics Research")[0]

    assert source.license == "MIT"

    assert source.citations[0].uri == "https://research.example.org/paper"
