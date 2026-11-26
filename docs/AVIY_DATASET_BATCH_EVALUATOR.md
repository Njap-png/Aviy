# Aviy Dataset Batch Evaluator

## Purpose

Runs the existing Aviy dataset evaluator against explicitly selected local
datasets. It does not discover or download datasets, register sources, approve
records for production, or authorize tokenizer training or pretraining.

## Manifest format

Paths are relative to the manifest file unless absolute paths are supplied.

```json
{
  "schema_version": "1.0",
  "datasets": [
    {
      "name": "english_pilot",
      "records": "english/records.jsonl",
      "metadata": "english/source_metadata.json"
    },
    {
      "name": "swahili_pilot",
      "records": "swahili/records.jsonl",
      "metadata": "swahili/source_metadata.json"
    }
  ]
}
```

Names must be unique and contain only letters, digits, underscores, and
hyphens. The first character must be a letter or digit.

## Run

```powershell
.\.venv\Scripts\python.exe -m aviy.corpus_governance.dataset_batch_evaluator `
  --manifest "PATH\batch_manifest.json" `
  --output "PATH\new_batch_output"
```

The output directory must not already exist. Each dataset receives its own
evaluator output directory. `batch_report.json` contains aggregate results,
per-dataset statuses, input hashes, report hashes, and staged/quarantined counts.

A failed dataset is recorded as FAILED while remaining datasets continue.
The command exits non-zero if any dataset fails.

## Governance boundary

`ACCEPTED_FOR_CORRECTED_STAGING` is a technical screening result only. It is
not legal clearance, source approval, record approval, or permission to train.
The production source registry and authorization settings remain unchanged.