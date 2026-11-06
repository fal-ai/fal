"""Benchmark runs for `fal dev app request --repeat`, saved and compared locally.

The box times each request (warmups first, then the measured runs) in one call,
so the numbers exclude the laptop's round trip to fal.run. A run saved with
``--name`` is a JSON file in ``~/.fal/dev/benchmarks/<session>/<name>.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

BENCHMARKS_DIR = Path("~/.fal/dev/benchmarks").expanduser()
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
# Fields that must match for a comparison to mean anything.
COMPARE_KEYS = ("app", "path", "method", "payload_sha256", "machine_type")
STAT_KEYS = ("median", "p90", "min", "max", "mean")


class BenchmarkError(Exception):
    pass


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile: always one of the measured values."""
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def stats(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p90": percentile(values, 0.9),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def payload_sha256(body: Any) -> str:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def source_digest(manifest: dict[str, str]) -> str:
    """A short id for the synced source tree, to tell code versions apart."""
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:12]


def check_name(name: str) -> str:
    if not NAME_RE.match(name):
        raise BenchmarkError(
            f"Invalid benchmark name {name!r}: use letters, digits, '.', '_' or '-'."
        )
    return name


def record(
    *,
    name: str | None,
    session: Any,
    path: str,
    body: Any,
    result: dict[str, Any],
    warmups: int,
) -> dict[str, Any]:
    runs = [run["elapsed_seconds"] for run in result["runs"]]
    return {
        "name": name,
        "created_at": time.time(),
        "session": session.name,
        "app": f"{Path(session.file_path).name}::{session.class_name}",
        "machine_type": session.machine_type,
        "method": "POST",
        "path": path,
        "payload_sha256": payload_sha256(body),
        "source": source_digest(session.manifest),
        "warmups": warmups,
        "repeat": len(runs),
        "runs": runs,
        "warmup_runs": [run["elapsed_seconds"] for run in result["warmup_runs"]],
        "stats": stats(runs),
    }


def path_for(session_name: str, name: str) -> Path:
    return BENCHMARKS_DIR / session_name / f"{check_name(name)}.json"


def save(run: dict[str, Any]) -> Path:
    target = path_for(run["session"], run["name"])
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(run, indent=2) + "\n")
    os.replace(tmp, target)
    return target


def load(ref: str, session_name: str | None) -> dict[str, Any]:
    """A saved run by name under the session's folder, or a ``.json`` file."""
    path = Path(ref)
    if session_name is not None and NAME_RE.match(ref):
        named = BENCHMARKS_DIR / session_name / f"{ref}.json"
        if named.is_file():
            path = named
    if not path.is_file() or path.suffix != ".json":
        where = f" for {session_name}" if session_name else ""
        raise BenchmarkError(f"No saved benchmark {ref!r}{where}.")
    try:
        run = json.loads(path.read_text())
    except ValueError as exc:
        raise BenchmarkError(f"{path} is not valid JSON: {exc}") from exc
    if not valid(run):
        raise BenchmarkError(f"{path} is not a saved benchmark.")
    return run


def valid(run: Any) -> bool:
    return (
        isinstance(run, dict)
        and all(key in run for key in ("name", "path", "repeat", "created_at"))
        and isinstance(run.get("stats"), dict)
        and all(isinstance(run["stats"].get(key), (int, float)) for key in STAT_KEYS)
    )


def saved(session_name: str) -> list[dict[str, Any]]:
    runs = []
    for path in sorted((BENCHMARKS_DIR / session_name).glob("*.json")):
        try:
            run = json.loads(path.read_text())
        except ValueError:
            continue
        if valid(run):
            runs.append(run)
    return sorted(runs, key=lambda run: run["created_at"])


def mismatches(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    return [
        f"{key}: {baseline.get(key)} vs {candidate.get(key)}"
        for key in COMPARE_KEYS
        if baseline.get(key) != candidate.get(key)
    ]


def _seconds(value: float) -> str:
    return f"{value * 1000:.1f}ms" if value < 1 else f"{value:.3f}s"


def _delta(before: float, after: float) -> str:
    delta = after - before
    sign = "+" if delta >= 0 else "-"
    text = f"{sign}{_seconds(abs(delta))}"
    if before:
        text += f" ({delta / before * 100:+.1f}%)"
    return text


def format_stats(values: dict[str, float]) -> str:
    return "  ".join(
        f"{key} {_seconds(values[key])}" for key in ("median", "p90", "min", "max")
    )


def format_runs(runs: list[float]) -> str:
    return " ".join(_seconds(value) for value in runs)


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """Rows of a side-by-side comparison, baseline first."""
    left, right = baseline["name"] or "baseline", candidate["name"] or "candidate"
    width = max(len(left), len(right), 9) + 2
    lines = [f"{'':8}{left:<{width}}{right:<{width}}change"]
    for key in STAT_KEYS:
        before, after = baseline["stats"][key], candidate["stats"][key]
        lines.append(
            f"{key:<8}{_seconds(before):<{width}}{_seconds(after):<{width}}"
            f"{_delta(before, after)}"
        )
    lines.append(
        f"{'runs':<8}{baseline['repeat']:<{width}}{candidate['repeat']:<{width}}"
    )
    lines.append(
        f"{'source':<8}{baseline.get('source', '?'):<{width}}"
        f"{candidate.get('source', '?'):<{width}}"
    )
    return lines
