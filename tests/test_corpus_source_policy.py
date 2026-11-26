from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def test_corpus_configuration_denies_training_authorization():
    text = (ROOT / "configs/data/corpus_config.yaml").read_text(encoding="utf-8")
    assert "training_authorized: false" in text
    assert "tokenizer_training_authorized: false" in text
    assert "pretraining_authorized: false" in text
    assert "production_training_authorized: false" in text


def test_production_source_registry_remains_empty():
    path = ROOT / "data/aviy_corpus/manifests/source_registry.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0"
    assert data["sources"] == []


def test_candidate_register_is_not_an_approval_registry():
    path = ROOT / "data/aviy_corpus/manifests/source_review_register.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["not_a_source_approval_registry"] is True
    assert data["training_authorized"] is False
    assert data["tokenizer_training_authorized"] is False
    assert len(data["sources"]) == 6


def test_no_candidate_has_training_permission():
    path = ROOT / "data/aviy_corpus/manifests/source_review_register.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert all(item["training_allowed"] is False for item in data["sources"])


def test_fineweb_pilot_hashes_match_register():
    path = ROOT / "data/aviy_corpus/manifests/source_review_register.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    for item in data["sources"]:
        sample = item.get("inspection_sample")
        if sample is None:
            continue
        file_path = ROOT / sample["path"]
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest().upper()
        assert digest == sample["sha256"]
        assert file_path.stat().st_size == sample["bytes"]


def test_review_report_has_no_unresolved_hash_placeholders():
    path = ROOT / "docs/PRETRAINING_SOURCE_REVIEW_2026-11-26.md"
    text = path.read_text(encoding="utf-8")
    assert "$manifestHash" not in text
    assert "System.Collections.Hashtable" not in text


def test_review_report_records_train_split():
    path = ROOT / "docs/PRETRAINING_SOURCE_REVIEW_2026-11-26.md"
    text = path.read_text(encoding="utf-8")
    assert re.search(r"(?m)^\s*Split inspected:\s*train only\s*$", text)
    assert not re.search(r"(?m)^\s*Split inspected:\s*rain only\s*$", text)


def test_candidate_register_is_separate_from_production_registry():
    doc = (ROOT / "docs/AVIY_CORPUS_SOURCE_REVIEW_REGISTER.md").read_text(
        encoding="utf-8"
    )
    assert "separately from the" in doc
    assert "NOT GRANTED" in doc
