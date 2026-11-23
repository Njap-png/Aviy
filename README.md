# Aviy

An independent, modular and evolving AI ecosystem.

- License: Apache-2.0
- Python: 3.12
- Deep learning: PyTorch
- Initial model: Aviy Base
- Initialization: Random weights; trained from scratch

## Development environment

Create and activate the virtual environment:

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1

Install the development dependencies:

    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"

Run verification:

    python -m pytest -q
    python -m ruff check aviy tests
    python -m pyright aviy tests
    python -m compileall -q aviy tests

See PROJECT_STATE.md and SESSION_HANDOFF.md before continuing development.

No pretrained weights may initialize Aviy Base.

## Engineering documentation

- [Engineering Standards](docs/ENGINEERING_STANDARDS.md)
- [Core Contracts](docs/CORE_CONTRACTS.md)
- [Architecture](ARCHITECTURE.md)
