# Dolly 15K ? Aviy instruction dataset

## Source

- Repository: databricks/databricks-dolly-15k
- Pinned revision: `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`
- Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k
- Language: English
- License: CC BY-SA 3.0 (`CC-BY-SA-3.0`)

## Attribution and license

This dataset is derived from Databricks Dolly 15K.
Attribute the source to Databricks, Inc. and the Dolly dataset
contributors. The dataset is distributed under the Creative
Commons Attribution-ShareAlike 3.0 license.

License details: https://creativecommons.org/licenses/by-sa/3.0/

When distributing adapted dataset material, comply with the
attribution and ShareAlike requirements.

## Preparation

- Original source rows: 15011
- Privacy-excluded rows: 16
- Duplicate rows removed by Aviy: 16
- Accepted records: 14979
- Seed: 42
- Governance audit: PASS

The complete source export is preserved in `source_records.jsonl`.
Privacy-excluded row indexes and detector categories are recorded in
`excluded_privacy_records.jsonl`; flagged source text is not copied
into the training splits.

## Splits

- Train: 13450
- Validation: 736
- Test: 793

## Important

This step prepares data only. It does not train or fine-tune a model.
