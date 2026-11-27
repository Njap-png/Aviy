# Aviy Original Corpus ? Source and Governance Policy

Version: 1.0-draft
Status: Proposed policy; source-specific approvals remain pending.

## 1. Core principle

Aviy may only collect and train on material whose provenance, permitted
use, privacy status and quality can be adequately established.

Availability on the internet does not mean permission to reuse material.

## 2. Source approval requirements

Before collecting a source beyond a bounded inspection, document:

1. Exact source name and owner.
2. Official source location.
3. Dataset or document revision.
4. License name and version.
5. Applicable terms of use.
6. Attribution requirements.
7. Share-alike or other downstream obligations.
8. Whether commercial use and model training are permitted.
9. Restrictions, exceptions and third-party content concerns.
10. Collection date and reproducibility information.

If the license or permission is ambiguous, place the source on hold.
Seek appropriate legal guidance where necessary.

A dataset-level license does not necessarily grant rights to every
individual document or item inside the dataset.

## 3. Preferred source hierarchy

Prioritize:

- Material written specifically for Aviy.
- Material created by authorized contributors.
- Verified public-domain educational resources.
- Reputable educational sources with compatible explicit licenses.
- Carefully evaluated open datasets with traceable provenance.

Third-party web crawls are not automatically approved.
Their underlying content rights must be examined.

## 4. Original Aviy-authored content

Original content should record:
- Authoring method.
- Creation date.
- Subject and category.
- Language and script.
- Reviewer.
- Factual verification status.
- Any external references used.

Human-written and AI-assisted content must be distinguishable.

AI-assisted material requires review. It must not be represented as
verified source material without verification.

## 5. Privacy and personal information

Do not intentionally collect:
- Passwords, credentials or authentication tokens.
- Private keys or secrets.
- Personal contact details without a valid basis.
- Private conversations or confidential documents.
- Sensitive personal records.
- Unnecessary personal identifiers.

Apply privacy checks during collection and again during final review.

Automated detection is imperfect. Names, addresses, indirect identifiers
and contextual personal information require human judgment.

Potentially sensitive records must be quarantined pending review.

Never print private source text or sensitive records in routine audit
logs and reports.

## 6. Content quality

Reject or quarantine material that is:
- Empty, corrupted or unreadable.
- Predominantly navigation or boilerplate.
- Excessively duplicated.
- Unintelligible or clearly malformed.
- Unreliable or materially misleading.
- Poorly matched to its claimed language.
- Unnecessarily promotional or spam-like.
- Missing essential provenance.

Technical and mathematical content should receive appropriate subject
review.

Swahili content intended to establish language quality requires review
by a native or highly proficient speaker.

## 7. Cybersecurity content safety

Prefer defensive and educational material:
- Secure coding.
- Network defense.
- Threat detection.
- Vulnerability management.
- Incident response.
- Security awareness.
- Authorized lab education.

Review potentially harmful operational instructions before inclusion.
Do not collect credentials, live secrets or private target information.

## 8. Collection and transformation

Every transformation must be reproducible and documented.

Preserve:
- Original source reference.
- Source revision.
- Collection timestamp.
- Original content hash where retention is permitted.
- Transformation script version.
- Output hash.
- Filtering and rejection reasons.

Do not silently rewrite source content and present it as original.

Any normalization, extraction, translation, summarization or synthetic
augmentation must be documented.

## 9. Quarantine and rejection

Quarantine when:
- Privacy indicators are detected.
- Licensing needs clarification.
- Language quality is uncertain.
- Content may be unsafe.
- Provenance is incomplete.
- Automated quality checks identify anomalies.

Reject when the source or content is clearly disallowed or unsuitable.

Do not use quarantined content in tokenizer training, pretraining,
instruction tuning or evaluation.

## 10. Approval authority

Approval must be explicit and recorded.

Source approval and record approval are separate decisions.

A source can be conditionally approved with restrictions, but only records
meeting those restrictions may be released.

No script should automatically convert an inspection sample into an
approved training dataset.

## 11. Audit and release

Every release must have:
- A source inventory.
- License and attribution register.
- Privacy report.
- Quality report.
- Duplicate report.
- Split-integrity report.
- Record-level approval manifest.
- SHA256 hashes.
- Version and change history.

A release must be reproducible from its approved sources and documented
processing steps.

## 12. Existing inspection pilots

FineWeb English and FineWeb-2 Swahili inspection pilots remain separate
research artifacts.

Their existence does not imply approval.
Their dataset-level license labels do not establish rights to every
underlying webpage.

They must not be merged into the production corpus without a separate,
documented source and record approval process.

## 13. Current policy status

This is a proposed governance policy. It does not grant legal permission,
approve any external source, or authorize training.
