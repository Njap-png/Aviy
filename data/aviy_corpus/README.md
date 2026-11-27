# Aviy Original Corpus

This directory is the proposed workspace for the original Aviy corpus.

No records are approved merely because they are placed here.

## Directory structure

- `original/` ? Aviy-authored source material and contributor submissions.
- `incoming/` ? newly collected material awaiting governance checks.
- `staging/` ? records undergoing normalization and quality processing.
- `quarantine/` ? records requiring rights, privacy, safety or quality review.
- `approved/` ? explicitly approved records, organized by release.
- `manifests/` ? source inventories, record approvals and dataset split manifests.
- `reports/` ? non-sensitive quality, provenance, privacy and reproducibility reports.
- `splits/train/` ? released training data.
- `splits/validation/` ? released validation data.
- `splits/test/` ? isolated final evaluation data.

## Rules

1. Do not place secrets, credentials or private records here.
2. Do not put unapproved records in training split directories.
3. Keep test data isolated from training and model selection.
4. Record source licenses, permissions and attribution.
5. Keep source text out of routine logs and reports.
6. Record hashes and transformation history.
7. Never overwrite a released corpus version silently.

The current directory structure is empty. Collection and training have not
been authorized by creating these directories.
