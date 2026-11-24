"""Evidence-based benchmark comparisons."""

import hashlib
import json
import math
import os
from pathlib import Path


def _reject_constant(value):
    raise ValueError(f"Non-standard JSON numeric constant: {value}")


def _read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_benchmarks(baseline, candidate, output):
    baseline = Path(baseline).expanduser().resolve()
    candidate = Path(candidate).expanduser().resolve()
    output = Path(output).expanduser()

    if not baseline.is_file() or not candidate.is_file():
        raise FileNotFoundError("Both benchmark inputs must be existing files.")

    output_resolved = output.resolve()
    if output_resolved in {baseline, candidate}:
        raise ValueError("Report output must not overwrite either benchmark input.")
    if output.is_symlink():
        raise ValueError("Report output cannot be a symlink.")
    if output.exists() and (
        os.path.samefile(output, baseline)
        or os.path.samefile(output, candidate)
    ):
        raise ValueError("Report output must not be a hard link to either input.")

    a = _read_json(baseline)
    b = _read_json(candidate)
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise TypeError("Benchmark files must contain JSON objects.")

    ma = a.get("metrics", a)
    mb = b.get("metrics", b)
    if not isinstance(ma, dict) or not isinstance(mb, dict):
        raise TypeError("Benchmark metrics must be JSON objects.")

    comparisons = {}
    for key in sorted(set(ma) & set(mb)):
        x, y = ma[key], mb[key]
        if (
            isinstance(x, (int, float)) and not isinstance(x, bool)
            and isinstance(y, (int, float)) and not isinstance(y, bool)
        ):
            if (
                (isinstance(x, float) and not math.isfinite(x))
                or (isinstance(y, float) and not math.isfinite(y))
            ):
                raise ValueError(f"Metric {key!r} must be finite.")
            try:
                difference = y - x
            except OverflowError as exc:
                raise ValueError(
                    f"Metric difference {key!r} exceeds supported numeric range."
                ) from exc
            if isinstance(difference, float) and not math.isfinite(difference):
                raise ValueError(f"Metric difference {key!r} is not finite.")
            comparisons[key] = {
                "baseline": x, "candidate": y, "difference": difference
            }

    report = {
        "baseline_sha256": _hash(baseline),
        "candidate_sha256": _hash(candidate),
        "comparisons": comparisons,
        "limitations": "Comparison only; no causal claim or automatic adoption.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
