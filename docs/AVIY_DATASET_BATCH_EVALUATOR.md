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

The batch is assembled in a temporary sibling directory. The final output
directory is published with a same-filesystem directory rename only after all
dataset attempts and the batch report have completed. An unexpected ordinary
failure cleans up the temporary directory and does not publish a partial final
output. If another process creates the destination before publication, that
destination is preserved and publication fails.

A hard process termination or power loss can leave an orphan directory named
`.OUTPUTNAME.aviy-staging-*` beside the requested output. The evaluator does
not automatically delete such directories because they may belong to another
active process. Inspect them and confirm no batch is running before manually
removing one.

A failed dataset is recorded as FAILED while remaining datasets continue.
Such a batch still publishes its complete report with status
`COMPLETED_WITH_FAILURES`. The command exits non-zero if any dataset fails.

## Governance boundary

`ACCEPTED_FOR_CORRECTED_STAGING` is a technical screening result only. It is
not legal clearance, source approval, record approval, or permission to train.
The production source registry and authorization settings remain unchanged.