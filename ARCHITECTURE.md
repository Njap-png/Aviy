# Aviy Architecture

## Architectural boundaries

1. Main intelligence system.
2. Autonomous controller.
3. Protected security and governance controller.
4. Independent recovery controller.
5. Audit and monitoring system.

## Core design

- Modular architecture registry.
- Replaceable model implementations.
- Independent learning and training engines.
- Hybrid memory and knowledge services.
- Specialist agent ecosystem.
- Protected authorization boundaries.

## Core contracts

`aviy.core.interfaces` defines the component lifecycle protocol.

`aviy.core.registry` provides validated named registrations.

`aviy.core.container` provides dependency injection with explicit
transient and singleton lifetimes.

`aviy.core.coordinator` manages component initialization, startup,
failure cleanup and reverse-order shutdown.

Detailed contracts are maintained in `docs/CORE_CONTRACTS.md`.

## Engineering standards

Package responsibilities, Python conventions, logging, errors and
testing are defined in `docs/ENGINEERING_STANDARDS.md`.

## Change policy

Significant production changes require evaluation, release gates and
appropriate authorization.

Update this document whenever verified architectural decisions change.
