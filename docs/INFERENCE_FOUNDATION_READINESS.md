# Aviy - Inference Foundation Readiness

Created UTC: 2026-11-23T11:52:09.195398+00:00

## Inspection results

- Python source files: 72
- Test files: 21
- Model-related source candidates: 42
- Configuration files: 6
- Checkpoint/tokenizer artifacts: 0

## Candidate source files

- `aviy\agents\runtime.py`
- `aviy\agents\windows_sandbox.py`
- `aviy\api\app.py`
- `aviy\autonomous\architecture_evolution.py`
- `aviy\autonomous\self_healing.py`
- `aviy\autonomous\self_improvement.py`
- `aviy\cli\main.py`
- `aviy\cognition\reasoning.py`
- `aviy\core\container.py`
- `aviy\core\coordinator.py`
- `aviy\core\errors.py`
- `aviy\core\interfaces.py`
- `aviy\core\registry.py`
- `aviy\data\audit.py`
- `aviy\data\pipeline.py`
- `aviy\evaluation\benchmarks.py`
- `aviy\inference\generation.py`
- `aviy\infrastructure\resources.py`
- `aviy\infrastructure\updates.py`
- `aviy\knowledge\research.py`
- `aviy\knowledge\retrieval.py`
- `aviy\learning\instruction.py`
- `aviy\math\numerical.py`
- `aviy\memory\store.py`
- `aviy\models\__init__.py`
- `aviy\models\architectures\__init__.py`
- `aviy\models\architectures\base.py`
- `aviy\models\architectures\families.py`
- `aviy\models\architectures\registry.py`
- `aviy\models\base_model.py`
- `aviy\models\specialists.py`
- `aviy\multimodal\pipelines.py`
- `aviy\security\authentication.py`
- `aviy\security\durable_governance.py`
- `aviy\security\governance.py`
- `aviy\tokenizer\__init__.py`
- `aviy\tokenizer\benchmark.py`
- `aviy\tokenizer\bpe.py`
- `aviy\training\benchmark.py`
- `aviy\training\engine.py`
- `aviy\training\losses.py`
- `aviy\training\optimizers.py`

## Blockers

- No model checkpoint or tokenizer artifact discovered.
- No common external tokenizer package is installed.

## Required inference contract

A verified inference interface must provide:

1. Explicit model/configuration loading.
2. Explicit tokenizer loading and compatibility validation.
3. Input validation and bounded context length.
4. Deterministic evaluation mode when requested.
5. Device selection with safe CPU fallback where supported.
6. Generation limits and cancellation/timeout boundaries.
7. Structured output, errors and reproducibility metadata.
8. Tests using a real compatible checkpoint, not fabricated weights.

## Safety and evidence boundary

This is a static readiness assessment. It does not claim that a model has been loaded, that inference works, or that capability benchmarks have been executed. Existing architectures and checkpoints must not be replaced without explicit approval.
