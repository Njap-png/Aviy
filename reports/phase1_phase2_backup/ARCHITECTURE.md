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

## Change policy

Significant production changes require release gates and appropriate authorization.

Update this document whenever verified architectural decisions change.

## Package responsibilities

The detailed package responsibility map and cross-package dependency
conventions are maintained in `docs/ENGINEERING_STANDARDS.md`.
Application packages should depend on stable public interfaces and avoid
circular imports.

## Shared engineering services

- `aviy.core.logging` provides standard application logger creation and
  startup logging configuration.
- `aviy.core.errors` provides the shared Aviy exception hierarchy.
- Engineering, testing, documentation, logging and error conventions are
  defined in `docs/ENGINEERING_STANDARDS.md`.

