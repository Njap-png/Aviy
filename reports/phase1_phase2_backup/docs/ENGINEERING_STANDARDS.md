# Aviy Engineering Standards

## 1. Package boundaries

The `aviy` package contains application code. Each package owns a distinct
responsibility and should expose stable interfaces rather than depend on
another package's private implementation details.

| Package | Responsibility |
|---|---|
| `core` | Shared interfaces, registries, foundational types, errors and logging |
| `models` | Model definitions and architecture implementations |
| `models/architectures` | Replaceable neural architecture implementations |
| `tokenizer` | Tokenization, vocabulary and tokenizer serialization |
| `data` | Dataset ingestion, validation, processing and provenance |
| `learning` | Learning algorithms and learning strategies |
| `training` | Training loops, optimization and checkpoint workflows |
| `cognition` | Reasoning, planning and cognitive coordination |
| `memory` | Working, episodic, semantic and other memory services |
| `knowledge` | Structured knowledge, retrieval and knowledge graphs |
| `agents` | Specialist agents and agent lifecycle |
| `tools` | Tool interfaces and controlled tool execution |
| `multimodal` | Image, audio, video and other modality services |
| `autonomous` | Autonomous research, experiments and controlled improvement |
| `evaluation` | Benchmarks, metrics and regression evaluation |
| `infrastructure` | Runtime, hardware, storage and infrastructure services |
| `security` | Authentication, authorization, privacy and security enforcement |
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
- Add tests for new behavior and regression tests for repaired behavior.
- Do not introduce dependencies without a demonstrated need.

## 3. Logging conventions

Use `aviy.core.logging.get_logger(__name__)` or an appropriately named
logger from the `aviy` namespace.

Configure logging at the application entry point, not independently in
every module. Use standard Python logging levels appropriately:

- DEBUG: diagnostic details useful during development.
- INFO: significant normal application events.
- WARNING: unexpected conditions that allow operation to continue.
- ERROR: an operation failed.
- CRITICAL: a severe condition requiring urgent attention.

Never log passwords, tokens, API keys, private credentials, sensitive
personal information, or unrestricted private user content. Avoid logging
raw datasets or model inputs when they may contain private information.
Do not use logs as a substitute for protected audit records.

## 4. Error conventions

Use the shared exception hierarchy in `aviy.core.errors` for expected
Aviy-specific failures:

- `AviyError`: common base exception.
- `ConfigurationError`: configuration failures.
- `ValidationError`: input or data validation failures.
- `ResourceLimitError`: resource quota or limit violations.
- `SecurityError`: security and authorization failures.
- `RecoveryError`: recovery and restoration failures.

Use standard Python exceptions where they are more appropriate. Raise
specific exceptions with actionable messages. Preserve the original cause
with exception chaining (`raise ... from exc`) when translating errors.
Do not silently swallow failures or expose secrets in exception messages.

## 5. Documentation conventions

Document each module's responsibility, public interfaces, configuration,
important constraints and examples where appropriate. Update architecture
and policy documents when verified decisions change. Distinguish implemented
and tested behavior from planned functionality.

## 6. Testing and quality gates

Run the project checks before accepting a milestone:

- `python -m pytest -q`
- `python -m ruff check aviy tests`
- `pyright aviy tests`
- `python -m compileall -q aviy tests`

Tests must be repeatable and should not require paid services or external
credentials unless a test explicitly documents that requirement.

## 7. Change and release safety

Keep experiments isolated from stable components. Preserve known-good
checkpoints and backups. Significant production changes require appropriate
evaluation, release gates and authorization. Never bypass security,
privacy, resource or recovery controls.
