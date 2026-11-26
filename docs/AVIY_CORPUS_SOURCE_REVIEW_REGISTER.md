# Aviy Candidate Source Review Register

Status: Research and review tracking only.
Training authorization: NOT GRANTED.
Tokenizer authorization: NOT GRANTED.

## Purpose

This register tracks candidate source investigations separately from the
production source registry. Presence here does not mean a source has been
registered, licensed, approved, or authorized for training.

The production registry remains:
`data/aviy_corpus/manifests/source_registry.json`

## Candidate decisions

| Candidate | Current state | Training permission |
|---|---|---|
| FineWeb English | Pending rights and human review | No |
| FineWeb-2 Swahili | Pending rights and human review | No |
| OpenWebMath | Candidate not reviewed | No |
| OWASP Cheat Sheet Series | Conditional candidate, not approved | No |
| The Stack v2 | Deferred | No |
| Dolly 15K | Separate instruction-stage review | No |

## FineWeb inspection evidence

The bounded English and Swahili samples remain inspection-only.
Their SHA256 values are recorded in the JSON register and dated review.

Hashes establish file integrity only. They do not establish legal rights,
factual accuracy, privacy clearance, or linguistic quality.

The inspected split was `train`. Provider test splits were not used.

## Required approval gates

1. Exact source and immutable revision.
2. Applicable license and terms.
3. Underlying content rights and third-party restrictions.
4. Attribution and notice preservation.
5. Privacy and sensitive information.
6. Language and subject-matter quality.
7. Reproducible collection and transformation.
8. Record-level review and approval.
9. Explicit source-level approval.
10. Explicit training-stage authorization.

A dataset-level license label is not sufficient to establish rights to
every underlying record.

## Restrictions

- Do not automatically register candidates in the production registry.
- Do not infer approval from inspection or integrity checks.
- Do not train a tokenizer or model on inspection pilots.
- Keep instruction-tuning data separate from base pretraining.
- Do not use quarantined or rejected records.
- This document is not legal advice.

## Current decision

No candidate in this register is approved for model training.
The production source registry must remain empty until a separately
authorized approval operation records an evidence-backed decision.
