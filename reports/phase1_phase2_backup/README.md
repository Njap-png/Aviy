# Aviy

An independent, modular and evolving AI ecosystem.

- License: Apache-2.0
- Python: 3.12
- Deep learning: PyTorch
- Initial model: Aviy Base
- Initialization: Random weights; trained from scratch

## Development

Activate the environment:

    .\.venv\Scripts\Activate.ps1

Install development dependencies:

    python -m pip install -e ".[dev]"

Run tests:

    python -m pytest

See PROJECT_STATE.md and SESSION_HANDOFF.md before continuing development.

No pretrained weights may initialize Aviy Base.

## Engineering standards

See [Engineering Standards](docs/ENGINEERING_STANDARDS.md) for package
responsibilities, Python conventions, logging, error handling and testing.
