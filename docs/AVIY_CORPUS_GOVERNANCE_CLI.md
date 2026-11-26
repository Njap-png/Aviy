# Aviy Corpus Governance CLI

Read-only commands for governed corpus inspection.

## Commands

- `aviy corpus sources`
- `aviy corpus validate --records PATH`
- `aviy corpus audit --records PATH`
- `aviy corpus manifest --manifest PATH`
- `aviy corpus verify --manifest PATH --records PATH`

Use `--root` to select a project root and `--registry` to select a source
registry. The default registry is
`data/aviy_corpus/manifests/source_registry.json`.

The records argument expects JSON Lines containing serialized governance
`CorpusRecord` objects. It does not automatically adapt generic
`DatasetRecord` objects from the separate data pipeline.

These commands do not register or approve sources, approve or quarantine
records, write manifests, modify records, or start training. Verification
rebuilds the expected manifest in memory and does not save it.

Privacy screening is conservative and is not a guarantee that all personal
information will be detected. Inspection does not grant training permission.
