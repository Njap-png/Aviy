# Aviy Inference Runtime

## Scope

Aviy now has a standalone inference runtime for `AviyBaseLM` and
`ByteBPETokenizer`. It reuses the existing autoregressive generation
implementation rather than introducing a second decoding algorithm.

## Runtime capabilities

- Construct an inference engine from a model and tokenizer.
- Validate tokenizer vocabulary against the model vocabulary.
- Validate prompt type, non-empty tokenization, and context length.
- Generate text using the existing `GenerationConfig` and `generate_tokens`.
- Export model weights and tokenizer into a standalone artifact directory.
- Load artifacts using PyTorch's `weights_only=True` loading mode.
- Verify SHA-256 hashes for model weights and tokenizer before loading.
- Inspect model/runtime metadata without exposing prompts or weights.

## Artifact contents

- `manifest.json`: artifact format, model configuration, and SHA-256 hashes.
- `model.pt`: CPU tensor state dictionary only.
- `tokenizer.json`: Aviy byte-level BPE tokenizer.

The export operation refuses to overwrite an existing destination. The
artifact hashes detect accidental or unauthorized file modification, but
they are not digital signatures and do not establish publisher identity.

## Verification boundary

Automated tests use a tiny randomly initialized model and a small synthetic
tokenizer corpus. They verify runtime behavior, artifact round-tripping,
compatibility checks, and integrity failure handling.

No trained Aviy checkpoint or production tokenizer was available during
this milestone. Consequently:

- This is an inference software foundation, not a trained AI release.
- Generated text from the smoke-test model is not meaningful capability
  evidence.
- No intelligence benchmark result is claimed.
- No model quality, safety, or production-readiness claim is made.
- A real model must be trained, evaluated, and approved before deployment.

## Device support

The runtime defaults to CPU. CUDA can be selected when PyTorch reports that
CUDA is available. This milestone does not claim GPU performance validation.
