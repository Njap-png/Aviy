# Aviy Phases 26-30 Implementation

## Phase 26 - API and CLI
FastAPI provides health, authenticated status, authenticated research requests
and OpenAPI documentation. Typer provides version, validate and loopback-only
serve commands. API authentication delegates to the existing
AuthenticationService. Configure AVIY_TOKEN_SIGNING_KEY with at least 32 bytes
and optionally AVIY_AUTH_DB before starting the API.

## Phase 27 - Interfaces
A responsive keyboard-accessible web interface is served through the API.
The desktop-compatible launcher opens loopback URLs only.

## Phase 28 - Deployment
The project uses its existing setuptools build configuration. Web assets are
included as package data. See ../deploy/README.md. A wheel build is not by
itself proof of a clean isolated installation.

## Phase 29 - Recovery
Backups contain SHA-256 manifests. Release archives can be restored into a
new empty directory. Production deployment still requires atomic release
switching, rollback drills and extended reliability testing.

## Phase 30 - Research
Benchmark comparisons record source hashes and metric differences. They do
not establish causality or authorize automatic model adoption.
