# Automated Dataset Evaluator

## Purpose

This module performs conservative automated technical screening of a locally
available JSONL dataset. It normalizes text, checks required metadata and
provenance, detects duplicate IDs and content, checks language/script fields,
and quarantines records that trigger basic privacy patterns.

It does not discover or download datasets.

## Required source metadata

Supply a JSON object containing:

- `source_id`
- `source_name`
- `source_uri`
- `revision`
- `license_id`
- `license_evidence_uri`
- `terms_uri`
- `attribution_text`
- `rights_scope`
- `underlying_rights_status`

The evaluator currently allows only CC0-1.0, CC-BY-4.0, CC-BY-SA-4.0,
ODC-BY-1.0, and PDDL-1.0. An allowlisted identifier is not proof that the
licence applies to every record or underlying third-party work.

`rights_scope` must be `dataset_and_contents` or `original_data`.
`underlying_rights_status` must be `covered` or `not_applicable`.

## Required record fields

Each JSONL record must contain:

- `record_id`
- `text`
- `language`
- `script`
- `source_id`
- `source_uri`
- `source_revision`
- `license_id`

## Run

```powershell
.\.venv\Scripts\python.exe -m aviy.corpus_governance.dataset_evaluator `
  --records "PATH\records.jsonl" `
  --metadata "PATH\source_metadata.json" `
  --output "PATH\evaluation_output"
```

Outputs:

- `corrected_accepted.jsonl`
- `quarantined.jsonl`
- `evaluation_report.json`

## Governance boundary

`ACCEPTED_FOR_CORRECTED_STAGING` means only that records passed this
automated screening and may enter a separate corrected staging area.

It does not mean legal rights were independently verified. It does not
register or approve a source in Aviy's production registry. It does not
approve records through the existing human-review governance engine.
It does not authorize tokenizer training or pretraining.

Privacy pattern detection is imperfect and cannot guarantee that all
personal or sensitive information has been found. The evaluator does not
replace rights review, privacy governance, attribution verification, or
the existing source and record approval process.

## Additional safety and audit behavior

The evaluator refuses to overwrite any of its four output files. Use a new
output directory for each evaluation run.

Record URLs are checked against the declared source URL using scheme, host,
effective port, and path boundaries. A different host or a similarly prefixed
path is not considered part of the source.

The output directory includes `attribution_notice.txt`, preserving the
attribution text supplied in source metadata. The report records the input
records SHA256, a deterministic hash of the metadata, hashes of the accepted,
quarantined, and attribution outputs, and the UTC evaluation time. Hashes
identify file content; they do not prove authenticity or legal rights.

Phone detection is heuristic. Requiring plausible digit counts and phone-style
formatting reduces some false positives, but privacy screening can still miss
personal information or flag non-phone text.

`underlying_rights_status: covered` is a declaration supplied by the dataset
operator. The evaluator does not verify that declaration, inspect legal
documents, obtain third-party permissions, or establish that underlying web
content is licensed. Technical staging acceptance is never legal clearance
or production approval.
