# Aviy Pretraining Source Review

**Review date:** 2026-11-26  
**Review status:** Pending  
**Training approval:** NOT APPROVED  
**Purpose:** Document the bounded FineWeb English and FineWeb-2 Swahili inspection pilots.

## 1. Executive decision

Both sources remain inspection-only. The bounded samples passed the recorded integrity and automated structural checks, but these results do not establish legal permission to train on the underlying webpage content, linguistic quality, factual accuracy, or complete privacy safety.

**Decision: Neither source is approved for model training.**

No tokenizer training, base-model pretraining, or model-selection activity may use these samples while approval remains pending.

## 2. Inspection scope and reproducibility

- FineWeb English repository: HuggingFaceFW/fineweb
- FineWeb English revision: 9bb295ddab0e05d785b879661af7260fed5140fc
- FineWeb English configuration: sample-10BT
- FineWeb-2 repository: HuggingFaceFW/fineweb-2
- FineWeb-2 revision: f9c13333eb981300149d5ca60a8e9d659b276b9
- FineWeb-2 configuration: swh_Latn
Split inspected: train only
- Maximum saved records per source: 50
- Maximum scanned records per source: 1,000
- Maximum output size per source: 10 MiB
- Source test splits used: No

Manifest SHA256:

F3BEE30AE7ADCD8BC2E29574051CD264F8E5FA03FA2C3DE6044E18D1FB285014

## 3. Sample integrity

| Sample | Records | Bytes | SHA256 |
|---|---:|---:|---|
| FineWeb English | 50 | 165,307 | 620916FD42BF84472ECD7E07C0A913FFB0348C7E5E924F7FD4B58B4DB0C9DE2C |
| FineWeb-2 Swahili | 50 | 212,248 | 1C5B4452B541D967B4A658897EF288D228AB444CC4ED02EFBAAFD7EBB07D9D50 |

Both sample hashes match the recorded expected hashes.

Both samples contain 50 valid JSON lines, 50 inspection-only records, and zero records marked approved for training.

## 4. FineWeb English findings

- Scanned: 54 records.
- Saved for inspection: 50 records.
- Rejected by privacy-pattern checks: 4 records.
- Detected rejection patterns: 2 phone-number flags and 2 email-address flags.
- Language classification: 50 of 50 records classified as English.
- Median text length: 1,493 characters.
- Median word count: 249.
- Valid source URLs: 50.
- Script metadata: not provided for all 50 records.
- Duplicate records: 0.
- Empty records: 0.
- Control-character flags: 0.
- High repeated-line-ratio flags: 0.
- Very-low-character-diversity flags: 0.
- HTML-marker flags: 0.

**Review limitation:** Missing script metadata must not be represented as source-provided metadata. Human review is still required.

## 5. FineWeb-2 Swahili findings

- Scanned: 50 records.
- Saved for inspection: 50 records.
- Language classification: 50 of 50 records classified as Swahili (swh).
- Script metadata: 50 of 50 records marked Latn.
- Median text length: 1,422 characters.
- Median word count: 216.
- Valid source URLs: 50.
- Duplicate records: 0.
- Empty records: 0.
- Control-character flags: 0.
- High repeated-line-ratio flags: 0.
- Very-low-character-diversity flags: 0.
- HTML-marker flags: 0.

### Source-domain concentration

The largest observed source-domain counts in the 50-record sample were:

| Domain | Records |
|---|---:|
| sw.wikipedia.org | 17 |
| www.ippmedia.com | 6 |
| www.unmultimedia.org | 5 |
| www.jamiiforums.com | 3 |

The remaining listed domains contributed one record each.

**Review limitation:** The sample is small. Human review must assess language quality, source diversity, suitability, and any applicable source-specific conditions.

## 6. Licensing and rights review

Both dataset cards identify ODC-BY 1.0 at the dataset/database level. That label alone does not establish that every individual webpage included in the crawled material is licensed for Aviy's intended use.

The applicable Common Crawl terms and underlying third-party content rights require separate review. The review must not assume that a webpage is unrestricted merely because it appears in a downloadable dataset.

Required before approval:

1. Review the applicable dataset license and notices.
2. Review the applicable Common Crawl terms.
3. Assess the rights status of underlying webpage content and relevant third-party restrictions.
4. Establish an attribution and notice-preservation plan.
5. Document any unresolved or unacceptable rights risks.
6. Obtain an appropriate legal review where necessary.

**Current rights decision: Pending. No legal clearance is claimed by this report.**

## 7. Privacy and quality limitations

Automated privacy-pattern checks are useful filters, not proof that personal information is absent. Human review and additional governance checks remain necessary.

Automated language classification and structural checks do not establish:

- Native-speaker linguistic quality.
- Factual correctness.
- Educational value.
- Absence of harmful or unsuitable material.
- Representative language or topic coverage.
- Permission to use individual source content for model training.

The four FineWeb English privacy-pattern rejections were excluded from the saved inspection sample. The original inspection samples must remain unchanged.

## 8. Approval gates

| Approval requirement | Status |
|---|---|
| Immutable dataset revisions recorded | Complete |
| Bounded inspection samples created | Complete |
| Sample hashes verified | Complete |
| Automated structural quality audit | Complete |
| English script metadata review | Pending |
| Human English-language review | Pending |
| Human Swahili-language review | Pending |
| Source diversity review | Pending |
| Privacy review beyond automated checks | Pending |
| Dataset and Common Crawl terms review | Pending |
| Underlying webpage rights review | Pending |
| Attribution plan | Pending |
| Final governance decision | Pending |

## 9. Source decisions

| Source | Decision | Reason |
|---|---|---|
| FineWeb English | PENDING — NOT APPROVED | Rights review, human review, and script-metadata limitation remain unresolved |
| FineWeb-2 Swahili | PENDING — NOT APPROVED | Rights review, human review, and source-diversity review remain unresolved |

## 10. Restrictions until approval

- Do not mark either source approved for training.
- Do not use either sample for tokenizer training or model training.
- Do not merge these samples with instruction-tuning datasets.
- Do not use a source test split for training or model selection.
- Preserve the existing sample files and manifest unchanged.
- Record any future approval or rejection with evidence and reasons.

## 11. Review conclusion

The bounded inspection achieved its technical purpose: immutable revisions were recorded, sample files were created within the specified limits, hashes were verified, and automated structural checks were completed.

It did not complete the legal, linguistic, privacy, or governance approval process.

**Final status: INSPECTION COMPLETE — TRAINING APPROVAL PENDING.**
