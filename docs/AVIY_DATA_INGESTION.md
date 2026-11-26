# Aviy Data Ingestion

## Purpose

The ingestion module provides two interfaces to one local intake service:
the Typer CLI and the authenticated FastAPI dashboard.

Supported formats:

- JSONL: one JSON object per non-empty line.
- JSON: an array of objects, an object with a `records` array, or one object.
- CSV: a header row containing a `text` column.
- TXT: one non-empty line per record.

All formats are normalized to JSONL for the existing dataset batch evaluator.

## CLI

```powershell
aviy ingest file --file .\dataset.csv --metadata .\source_metadata.json
aviy ingest history
```

Optional parameters: `--name` and `--format jsonl|json|csv|txt`.

Metadata must satisfy the existing evaluator's complete source metadata
requirements, plus `language` and `script`.

## Dashboard

Start Aviy with its existing authenticated local API:

```powershell
$env:AVIY_TOKEN_SIGNING_KEY = "<your existing signing key>"
aviy serve
```

Open `http://127.0.0.1:8000/`. Authenticate with a bearer token and use
the Data Ingestion section. The browser sends file contents to the local API
as base64-encoded JSON; uploads are limited to 50 MiB.

API endpoints:

- `POST /v1/ingestion/intake`
- `GET /v1/ingestion/history`

Both require the existing bearer-token authentication.

## Storage

Each intake receives a UUID directory under `data/ingestion/runs/`.
The original file is preserved under `input/`, alongside normalized JSONL,
the supplied metadata, a generated batch manifest and an ingestion report.
The existing batch evaluator publishes its own results under `evaluation/`.
A local append-only JSONL history is stored at `data/ingestion/history.jsonl`.

## Security and reliability safeguards

The intake service enforces a 50 MiB decoded file limit and validates UTF-8,
supported formats, record structure, dataset names and filenames. Path
separators, drive-qualified paths, control characters and reserved Windows
device names are rejected.

The API bounds the streamed request body and Base64 field before decoding,
validates optional field types, and retains the existing bearer-token
authentication requirement. Both ingestion API endpoints remain authenticated.

A storage lock coordinates local intake and history operations across threads
and processes. Identical submissions with the same filename, format, dataset
name, content and metadata reuse the existing ingestion result rather than
creating a second run. The lock is held through evaluation and history
publication, so concurrent identical submissions are serialized.

JSON writes use unique temporary files and atomic replacement. History writes
are serialized and flushed. Failed intake publication removes its incomplete
run directory. An evaluator failure is recorded as a failed evaluation and
does not authorize training.

These safeguards do not make the local API suitable for exposure to the public
internet. Keep it bound to loopback, use the existing authentication system,
and protect the host and its storage directory.

## Governance boundaries

Ingestion is not source approval, record approval, legal clearance,
tokenizer authorization or pretraining authorization.

The existing dataset evaluator and batch evaluator remain authoritative.
No production source registry is modified. No authorization flags are changed.
No training is started. Metadata rights claims are not independently verified.
The 50 MiB limit is a local intake safeguard, not a privacy or legal guarantee.

TXT conversion creates record identifiers and provenance fields from the
explicitly supplied metadata where those fields are absent. Existing conflicting
record fields are preserved for the evaluator to assess.
