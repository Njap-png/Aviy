# Phase 9: Google Colab GPU smoke test

This procedure verifies that Aviy can train on a real CUDA GPU, use supported
mixed precision, report GPU memory, save a checkpoint, and resume from it.
It is a small infrastructure test, not full language-model training.

## 1. Start a GPU runtime

1. Open Google Colab.
2. Create a new notebook.
3. Select **Runtime > Change runtime type**.
4. Select **T4 GPU** (or another available CUDA GPU).
5. Save the runtime settings.

## 2. Clone the repository

Run:

```python
!git clone https://github.com/Njap-png/Aviy.git
%cd Aviy
```

If the repository is private, authenticate with GitHub using Colab's supported
authentication flow. Never paste a personal access token into a notebook cell
or commit it to the repository.

## 3. Install Aviy

```python
!python -m pip install --upgrade pip
!python -m pip install -e .
```

If installation reports missing project dependencies, resolve those from the
project's declared dependency files rather than installing arbitrary packages.

## 4. Confirm CUDA is available

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("Select a CUDA GPU runtime before continuing.")

print("GPU:", torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())
```

## 5. Run the training, checkpoint, and resume smoke test

Automatic precision selects the engine's automatic CUDA precision behavior:

```python
!python scripts/colab_gpu_smoke.py --precision auto
```

For an explicit FP16 test:

```python
!python scripts/colab_gpu_smoke.py --precision fp16
```

If the GPU reports BF16 support, also run:

```python
!python scripts/colab_gpu_smoke.py --precision bf16
```

The script exits with an error if CUDA is unavailable, training produces
non-finite losses, checkpoint resume fails, or required GPU resource metrics
are missing.

## 6. Preserve the report

The default report is written to:

`/content/aviy_phase9_gpu_smoke.json`

The checkpoint is written beside it. Download or copy the JSON report to
Google Drive if you want to preserve the evidence after the Colab runtime ends.
Do not commit temporary model checkpoints to Git.

## What a passing run proves

- The selected runtime exposes a CUDA GPU to PyTorch.
- Aviy performs actual training steps on CUDA.
- The requested precision mode completes with finite losses.
- A checkpoint can be saved and resumed.
- The resumed global step is correct.
- CUDA device and memory resource metrics are reported.

A passing smoke test does not prove large-scale training stability, model
quality, or improved language capabilities.
