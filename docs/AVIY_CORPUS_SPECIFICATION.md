# Aviy Original Corpus ? Specification

Version: 1.0-draft
Status: Architecture defined; corpus collection and training approval pending.

## 1. Purpose

Aviy Original Corpus is a controlled, multilingual, educational and technical
pretraining corpus developed for the Aviy language model.

The objective is not to maximize raw document volume. It is to build a
reproducible collection of useful, coherent, sufficiently accurate and
properly authorized text.

The corpus must support:
- General English comprehension and communication.
- University-level mathematics and physics.
- Programming and computer science.
- Defensive cybersecurity and secure software engineering.
- Natural Swahili and multilingual understanding.
- Scientific and technical reasoning.

This specification does not authorize training on any existing pilot dataset.

## 2. Proposed content distribution

Initial planning targets, measured by eligible training tokens:

| Category | Target |
|---|---:|
| General English and educational material | 25% |
| Mathematics and physics | 20% |
| Programming and computer science | 20% |
| Defensive cybersecurity | 15% |
| Swahili | 15% |
| Other sciences | 5% |

These are planning targets, not mandatory quotas. They may be revised
after measuring corpus quality, language coverage and tokenizer behavior.

A category must never be filled with low-quality material merely to reach
its target.

## 3. Content categories

### 3.1 General English
- Natural written English.
- Educational explanations.
- Reading comprehension.
- General science and factual knowledge.
- Grammar and vocabulary.
- Everyday communication.

### 3.2 Mathematics and physics
- Arithmetic and number theory.
- Algebra and geometry.
- Calculus and linear algebra.
- Probability and statistics.
- Mechanics and electromagnetism.
- Step-by-step derivations with verified results.

### 3.3 Programming and computer science
- Python and JavaScript.
- Algorithms and data structures.
- Software engineering.
- Databases and operating systems.
- Networking and computer architecture.
- Secure coding and documented examples.

Code must be checked for syntax and quality where practical. Never execute
untrusted source code as part of ordinary dataset validation.

### 3.4 Cybersecurity
- Networking fundamentals.
- Defensive security operations.
- Threat identification and incident response.
- Secure configuration.
- Vulnerability concepts and responsible remediation.
- Defensive analysis and secure development.

Content must be reviewed for safety and must not become operational
instructions for unauthorized compromise.

### 3.5 Swahili
- Natural, idiomatic Swahili.
- Grammar and vocabulary.
- Educational and explanatory prose.
- Science and technical terminology.
- Multiple topics and writing styles.

Language metadata or automated language identification alone does not
establish linguistic quality. Native or highly proficient review is
required for source approval.

### 3.6 Other sciences
- Biology, chemistry and earth science.
- Scientific literacy.
- Evidence-based explanations.
- Carefully reviewed educational material.

## 4. Source classes

Every corpus record must belong to a documented source class:

1. Original Aviy-authored material.
2. Public-domain material with verified status.
3. Third-party material with a compatible, documented license.
4. Explicitly authorized material with recorded permission.

Unverified web content is not an approved source class.

AI-assisted drafts must be identified as such and reviewed for accuracy,
duplication, provenance and rights before approval. Generated text is not
automatically factual or suitable for training.

## 5. Quality requirements

Every candidate record must be evaluated for:

- Non-empty, readable text.
- Language and script consistency.
- Useful information density.
- Coherence and grammatical quality.
- Repetition and boilerplate.
- Duplicate and near-duplicate content.
- Factual and technical reliability.
- Source traceability.
- Privacy and personal information.
- Licensing and permission.
- Safety suitability.

Quality decisions must be recorded at record level.

A heuristic score cannot replace human review where human review is required.

## 6. Record-level governance

Each record must have a stable identifier and provenance metadata.

Required fields:
- record_id
- text
- language
- script
- category
- source_type
- source_name
- source_uri or documented original-authoring reference
- source_revision, where applicable
- license
- attribution requirements
- collected_at or authored_at
- content_sha256
- review_status
- privacy_status
- approval_status

Optional fields may include:
- author or organization
- publication date
- reviewer
- review notes
- technical validation results
- parent document identifier
- transformation history

Never store credentials, access tokens or unrelated private information
in corpus records.

## 7. Review and approval states

Allowed record states:

- collected
- pending_review
- human_review_required
- quarantine_candidate
- quarantined
- rejected
- approved

Only records with explicit approved status and complete provenance may
enter an approved training manifest.

A source-level approval does not automatically approve every record.

Quarantined and rejected records must never enter training manifests.

## 8. Dataset splitting

The corpus must maintain distinct training, validation and test manifests.

Rules:
- Test data is reserved for final evaluation.
- Test data must never be used for gradient updates.
- Test data must never guide checkpoint selection or tuning.
- Related documents and duplicate content must not leak across splits.
- Split assignment must be deterministic and reproducible.
- Split manifests must record their source dataset version and hashes.

Validation data may be used for controlled model selection.
Test data remains untouched until final evaluation.

## 9. Tokenizer policy

The production tokenizer may only be trained on an explicitly approved
training corpus manifest.

Before tokenizer training:
- Source and record approvals must be complete.
- Language and script coverage must be measured.
- Duplicate and privacy checks must pass.
- Train/validation/test isolation must be verified.
- The corpus version and manifest hashes must be recorded.

Measure:
- Vocabulary coverage.
- Token fertility by language.
- Unknown or fallback behavior.
- Round-trip behavior.
- Encoding and decoding correctness.
- English and Swahili performance separately.

A tokenizer benchmark is not proof that the underlying corpus is suitable
for model training.

## 10. Reproducibility

Every approved corpus release must contain:
- Corpus version.
- Collection and transformation scripts.
- Source inventory.
- Per-source revisions and licenses.
- Record counts.
- Split counts.
- Content hashes.
- Quality and governance reports.
- Known limitations.
- Approval history.

Never silently replace a released corpus.

## 11. Release gates

A corpus release is blocked if:
- Licensing or permission is unresolved.
- Required provenance is missing.
- Privacy review is incomplete.
- Duplicate or split leakage checks fail.
- Required language review is incomplete.
- Quality thresholds are not met.
- Training manifests include unapproved records.

## 12. Initial status

Architecture: defined by this draft.
Collection: not started.
Approved production records: none established by this document.
Production tokenizer trained on this corpus: no.
Base-model training authorized by this document: no.

This document defines policy and architecture. It is not a training
authorization or legal opinion.
