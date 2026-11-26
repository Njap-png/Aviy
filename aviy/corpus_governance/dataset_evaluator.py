"""Conservative automated evaluation of locally supplied dataset JSONL files.

This evaluator produces corrected staging files only. It never approves a
production source, authorizes tokenizer training, or authorizes pretraining.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit


ALLOWED_LICENSES = {
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "ODC-BY-1.0",
    "PDDL-1.0",
}

REQUIRED_METADATA = (
    "source_id",
    "source_name",
    "source_uri",
    "revision",
    "license_id",
    "license_evidence_uri",
    "terms_uri",
    "attribution_text",
    "rights_scope",
    "underlying_rights_status",
)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
SECRET_RE = re.compile(
    r"(?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*\S+",
    re.I,
)
PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")


def normalize_text(value: str) -> str:
    """Normalize Unicode and collapse whitespace."""
    value = unicodedata.normalize("NFC", value)
    return " ".join(value.split())


def valid_http_url(value: object) -> bool:
    """Accept absolute HTTP(S) URLs with a host and no embedded credentials."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = urlsplit(value.strip())
        return (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and parsed.port != 0
        )
    except ValueError:
        return False


def uri_belongs_to_source(record_uri: object, source_uri: object) -> bool:
    """Compare URL origin and path boundaries rather than string prefixes."""
    if not valid_http_url(record_uri) or not valid_http_url(source_uri):
        return False
    try:
        record = urlsplit(record_uri.strip())
        source = urlsplit(source_uri.strip())

        def origin(parsed):
            scheme = parsed.scheme.lower()
            host = parsed.hostname.lower().rstrip(".")
            port = parsed.port
            if port is None:
                port = 443 if scheme == "https" else 80
            return scheme, host, port

        if origin(record) != origin(source):
            return False

        if re.search(r"%2f|%5c", source.path + record.path, re.I):
            return False

        base_path = unquote(source.path or "/")
        candidate_path = unquote(record.path or "/")
        if "\\\\" in base_path or "\\\\" in candidate_path:
            return False

        base_path = "/" + "/".join(x for x in base_path.split("/") if x)
        candidate_path = "/" + "/".join(x for x in candidate_path.split("/") if x)
        if base_path == "/":
            return True
        return candidate_path == base_path or candidate_path.startswith(base_path + "/")
    except (ValueError, AttributeError):
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_source_metadata(metadata: object) -> list[str]:
    """Return source-level blockers. This does not verify legal rights."""
    issues: list[str] = []

    if not isinstance(metadata, dict):
        return ["Source metadata must be a JSON object."]

    for field in REQUIRED_METADATA:
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"Missing or invalid metadata field: {field}")

    for field in ("source_uri", "license_evidence_uri", "terms_uri"):
        if field in metadata and not valid_http_url(metadata[field]):
            issues.append(f"{field} must be an HTTP(S) URL.")

    if metadata.get("license_id") not in ALLOWED_LICENSES:
        issues.append("Licence is not in the evaluator's explicit allowlist.")

    if metadata.get("rights_scope") not in {
        "dataset_and_contents",
        "original_data",
    }:
        issues.append("Rights scope is not explicitly covered.")

    if metadata.get("underlying_rights_status") not in {
        "covered",
        "not_applicable",
    }:
        issues.append("Underlying content rights are unresolved.")

    return issues


def privacy_issues(text: str) -> list[str]:
    """Conservative pattern-based privacy screening; not a guarantee."""
    issues = []
    if EMAIL_RE.search(text):
        issues.append("possible_email")
    for match in PHONE_RE.finditer(text):
        candidate = match.group(0)
        digit_count = sum(character.isdigit() for character in candidate)
        formatted = candidate.startswith("+") or any(
            character in " ()-." for character in candidate
        )
        if 8 <= digit_count <= 15 and formatted:
            issues.append("possible_phone_number")
            break
    if SECRET_RE.search(text):
        issues.append("possible_secret")
    if PRIVATE_KEY_RE.search(text):
        issues.append("private_key_marker")
    return issues


def evaluate_records(
    records_path: Path,
    metadata: dict,
    output_dir: Path,
    minimum_score: float = 70.0,
    minimum_characters: int = 40,
) -> dict:
    """Evaluate JSONL records and write corrected staging/quarantine outputs."""
    records_path = Path(records_path)
    output_dir = Path(output_dir)
    if not records_path.is_file():
        raise FileNotFoundError(f"Records file does not exist: {records_path}")

    output_names = (
        "corrected_accepted.jsonl",
        "quarantined.jsonl",
        "evaluation_report.json",
        "attribution_notice.txt",
    )
    for name in output_names:
        target = output_dir / name
        if target.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing evaluator output: {target}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_issues = evaluate_source_metadata(metadata)
    accepted: list[dict] = []
    quarantined: list[dict] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    total = 0

    with records_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue

            total += 1
            issues: list[str] = []

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                quarantined.append({
                    "line_number": line_number,
                    "issues": ["Invalid JSON."],
                })
                continue

            if not isinstance(record, dict):
                quarantined.append({
                    "line_number": line_number,
                    "issues": ["Record must be a JSON object."],
                })
                continue

            record = dict(record)
            record_id = record.get("record_id")
            text = record.get("text")

            if not isinstance(record_id, str) or not record_id.strip():
                issues.append("Missing record_id.")
            elif record_id in seen_ids:
                issues.append("Duplicate record_id.")
            else:
                seen_ids.add(record_id)

            if not isinstance(text, str):
                issues.append("Text must be a string.")
                normalized = ""
            else:
                normalized = normalize_text(text)
                if len(normalized) < minimum_characters:
                    issues.append("Text is shorter than the minimum length.")
                if normalized != text:
                    record["text"] = normalized

            if normalized:
                fingerprint = hashlib.sha256(
                    normalized.encode("utf-8")
                ).hexdigest()
                if fingerprint in seen_hashes:
                    issues.append("Duplicate normalized text.")
                else:
                    seen_hashes.add(fingerprint)

                issues.extend(privacy_issues(normalized))

            for field in (
                "language",
                "script",
                "source_id",
                "source_uri",
                "source_revision",
                "license_id",
            ):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    issues.append(f"Missing or invalid record field: {field}")

            if record.get("source_id") != metadata.get("source_id"):
                issues.append("Record source_id does not match source metadata.")

            if record.get("source_revision") != metadata.get("revision"):
                issues.append("Record revision does not match source metadata.")

            if record.get("license_id") != metadata.get("license_id"):
                issues.append("Record licence does not match source metadata.")

            record_uri = record.get("source_uri")
            base_uri = metadata.get("source_uri", "")
            if not valid_http_url(record_uri):
                issues.append("Record source_uri must be an HTTP(S) URL.")
            elif not uri_belongs_to_source(record_uri, base_uri):
                issues.append("Record URI is outside the declared source URI.")

            if source_issues:
                issues.extend(source_issues)

            if issues:
                quarantined.append({
                    "line_number": line_number,
                    "record_id": record_id,
                    "issues": sorted(set(issues)),
                    "record": record,
                })
            else:
                accepted.append(record)

    score = (len(accepted) / total * 100.0) if total else 0.0
    dataset_gate_passed = (
        not source_issues
        and total > 0
        and len(accepted) > 0
        and score >= minimum_score
    )

    if not dataset_gate_passed and accepted:
        for record in accepted:
            quarantined.append({
                "record_id": record.get("record_id"),
                "issues": ["Dataset-level acceptance threshold or source gate failed."],
                "record": record,
            })
        accepted = []

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    accepted_path = output_dir / "corrected_accepted.jsonl"
    quarantined_path = output_dir / "quarantined.jsonl"
    attribution_path = output_dir / "attribution_notice.txt"
    report_path = output_dir / "evaluation_report.json"

    write_jsonl(accepted_path, accepted)
    write_jsonl(quarantined_path, quarantined)

    attribution_text = metadata.get("attribution_text", "")
    attribution_path.write_text(
        str(attribution_text).strip() + "\n",
        encoding="utf-8",
    )

    report = {
        "status": (
            "ACCEPTED_FOR_CORRECTED_STAGING"
            if dataset_gate_passed
            else "HOLD_NOT_ACCEPTED"
        ),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_records_sha256": sha256_file(records_path),
        "source_metadata_sha256": hashlib.sha256(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "records_total": total,
        "records_staged": len(accepted),
        "records_quarantined": len(quarantined),
        "quality_score_percent": round(score, 2),
        "minimum_score_percent": minimum_score,
        "minimum_characters": minimum_characters,
        "source_metadata_issues": source_issues,
        "attribution_text_preserved": bool(str(attribution_text).strip()),
        "output_sha256": {
            "corrected_accepted.jsonl": sha256_file(accepted_path),
            "quarantined.jsonl": sha256_file(quarantined_path),
            "attribution_notice.txt": sha256_file(attribution_path),
        },
        "legal_rights_independently_verified": False,
        "production_source_registry_changed": False,
        "tokenizer_training_authorized": False,
        "pretraining_authorized": False,
        "training_started": False,
        "note": (
            "Staging acceptance is an automated technical screening result, "
            "not legal clearance, production source approval, or training authorization."
        ),
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conservatively evaluate a local JSONL dataset."
    )
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-score", type=float, default=70.0)
    parser.add_argument("--minimum-characters", type=int, default=40)
    args = parser.parse_args()

    if not args.records.is_file() or not args.metadata.is_file():
        parser.error("Records and metadata must both be existing files.")
    if not 0 <= args.minimum_score <= 100:
        parser.error("--minimum-score must be between 0 and 100.")
    if args.minimum_characters < 1:
        parser.error("--minimum-characters must be positive.")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    report = evaluate_records(
        args.records,
        metadata,
        args.output,
        args.minimum_score,
        args.minimum_characters,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
