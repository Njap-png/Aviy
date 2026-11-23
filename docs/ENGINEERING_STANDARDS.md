# Aviy Engineering Standards

## 1. Package boundaries

The `aviy` package contains application code. Each package owns a distinct
responsibility and exposes stable public interfaces.

| Package | Responsibility |
|---|---|
| `core` | Interfaces, registries, dependency injection, coordination, errors and logging |
| `models` | Model definitions and architecture implementations |
| `models/architectures` | Replaceable neural architecture implementations |
| `tokenizer` | Tokenization and vocabulary |
| `data` | Dataset ingestion, validation, processing and provenance |
| `learning` | Learning algorithms and strategies |
| `training` | Training loops, optimization and checkpoint workflows |
| `cognition` | Reasoning, planning and cognitive coordination |
| `memory` | Memory services |
| `knowledge` | Structured knowledge and retrieval |
| `agents` | Specialist agents and agent lifecycle |
| `tools` | Tool interfaces and controlled execution |
| `multimodal` | Image, audio, video and modality services |
| `autonomous` | Autonomous research and controlled improvement |
| `evaluation` | Benchmarks, metrics and regression evaluation |
| `infrastructure` | Runtime, hardware and storage |
| `security` | Authentication, authorization, privacy and enforcement |
| `api` | External application programming interfaces |
| `cli` | Command-line interface |

Cross-package functionality should use documented public interfaces.
Avoid circular imports. Keep security and authorization enforcement
independent of ordinary agent or model decisions.

## 2. Python and code conventions

- Target Python 3.12.
- Use type annotations on public interfaces.
- Prefer small, focused modules and explicit dependencies.
- Keep configuration separate from implementation.
- Avoid hidden global state and circular imports.
- Add tests for new behavior and repaired behavior.
- Do not introduce dependencies without demonstrated need.

## 3. Logging conventions

Use `aviy.core.logging.get_logger(__name__)`.

Configure logging at the application entry point. Use standard Python
logging levels appropriately:

- DEBUG: diagnostic details.
- INFO: significant normal events.
- WARNING: unexpected conditions that allow operation to continue.
- ERROR: failed operations.
- CRITICAL: severe conditions requiring urgent attention.

Never log passwords, tokens, API keys, credentials, sensitive personal
information or unrestricted private user content.

## 4. Error conventions

Use the shared exception hierarchy in `aviy.core.errors`:

- `AviyError`: common base.
- `ConfigurationError`: configuration failures.
- `ValidationError`: input or data validation failures.
- `ResourceLimitError`: resource quota violations.
- `SecurityError`: security and authorization failures.
- `RecoveryError`: recovery failures.
- `RegistryError`: registry failures.
- `DependencyResolutionError`: dependency resolution failures.
- `LifecycleError`: component lifecycle failures.

Use standard Python exceptions where appropriate. Preserve original causes
with exception chaining when translating errors.

## 5. Documentation conventions

Document module responsibilities, public interfaces, configuration,
important constraints and examples. Distinguish tested functionality
from planned functionality.

## 6. Testing and quality gates

Run:

- `python -m pytest -q`
- `python -m ruff check aviy tests`
- `python -m pyright aviy tests`
- `python -m compileall -q aviy tests`

Tests should be repeatable and should not require paid services or
external credentials unless explicitly documented.

## 7. Change and release safety

Keep experiments isolated from stable components. Preserve known-good
backups. Significant production changes require evaluation and
appropriate authorization. Never bypass security, privacy, resource
or recovery controls.
