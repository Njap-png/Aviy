# Aviy Original Corpus ? Governance Engine

Version: 1.0 draft

## Purpose

The independent `aviy.corpus_governance` package provides source-level
and record-level governance for Aviy Original Corpus.

It does not replace the existing general dataset pipeline.

## Source approval

A source must have:
- A stable source ID and name.
- A source URI and revision.
- A documented license.
- Explicitly approved license status.
- Explicitly approved source status.
- Explicit training permission.
- Reviewer identity and approval timestamp.
- Attribution requirements.

An empty source registry means no third-party source is approved.

## Record approval

Records carry language, script, category, source provenance, revision,
license, attribution, group identity, content hash and review states.

`approve_record` requires explicit human review and a reviewer identity.
Automated checks cannot approve records.

Privacy pattern checks are conservative screening only. They cannot prove
that text contains no personal information. Human review remains required.

## Manifest generation

`build_manifest` accepts only records that:
- Have explicit approved status.
- Have human review evidence.
- Have clear privacy status.
- Match an approved source and license.
- Pass provenance and hash checks.
- Contain no detected privacy patterns.
- Are not duplicates.

The manifest stores record metadata and hashes, not source text.

Related records sharing a group ID are assigned to the same split.
The split is deterministic for a fixed seed and group ID.

## Training access

`training_records` returns only entries assigned to the training split.
Validation and test records are never returned by this accessor.

Test data must remain reserved for final evaluation and cannot be used
for gradient updates or model selection.

## Important limitations

- Regex privacy screening is not a complete privacy review.
- A registered license does not constitute legal advice.
- A source must be reviewed by an authorized person before approval.
- Hashes prove content integrity, not factual correctness or ownership.
- Human language and technical quality review remain necessary.
- This package does not start training or approve any source by itself.

## Initial state

The source registry is intentionally empty.
No source or record is approved by installation of this engine.
No tokenizer or model training is authorized by this milestone.
