"""Internal helpers shared by `apps.app_gpus` and `runners.runners_gpus`.

Both rely on the same `/applications/metrics` payload and the same GPU name
normalization. Living in their own module avoids a private cross-import
between sibling modules.
"""

from __future__ import annotations

import re
from http import HTTPStatus
from typing import TYPE_CHECKING

import httpx

import fal.flags as flags

if TYPE_CHECKING:
    from .client import SyncServerlessClient


# Ports the dashboard's GPU name normalization. SKU rules collapse driver
# variants (PCIE/SXM/MIG, etc.) into a single short name, e.g.
# "NVIDIA H100 80GB HBM3" and "NVIDIA H100 80GB HBM3 MIG 3g.40gb" → "H100".
_CPU_MACHINE_TYPES = {"M", "L", "S", "XS", "XL"}

_GPU_NAME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(RTX.?5090|GPU-RTX5090)\b", re.IGNORECASE), "5090"),
    (re.compile(r"\b(A100|GPU-A100)\b", re.IGNORECASE), "A100"),
    # L40S is a distinct GPU but the frontend collapses it into "L40"; keep
    # parity until the dashboard distinguishes them.
    (re.compile(r"\b(L40S?|GPU-L40)\b", re.IGNORECASE), "L40"),
    (re.compile(r"\b(H100|GPU-H100)\b", re.IGNORECASE), "H100"),
    (re.compile(r"\b(H200|GPU-H200)\b", re.IGNORECASE), "H200"),
    (re.compile(r"\b(B200|GPU-B200)\b", re.IGNORECASE), "B200"),
]

_RTX_RE = re.compile(r"RTX\s*(?:PRO\s*)?([A-Z]?\d{4}(?:\s+(?:Ti|Super|SUPER|Ada|XT))?)")
_MIG_RE = re.compile(r"\b([A-Z]\d{3,4})\b.*\bMIG\b")
_MODEL_RE = re.compile(r"\b([A-Z]\d{2,4}\w?)\b")
# Ordered after _MODEL_RE: AMD MI\d names like "MI300X" can't satisfy the
# generic [A-Z]\d{2,4} model pattern (no \b between M and I), so it's safe to
# fall through to the MI matcher.
_MI_RE = re.compile(r"\b(MI\d{3}\w*)\b")


def _shorten_gpu_name(name: str) -> str:
    if name in _CPU_MACHINE_TYPES:
        return ""
    m = _RTX_RE.search(name)
    if m:
        return f"RTX {m.group(1)}"
    m = _MIG_RE.search(name)
    if m:
        return f"{m.group(1)} MIG"
    m = _MODEL_RE.search(name)
    if m:
        return m.group(1)
    m = _MI_RE.search(name)
    if m:
        return m.group(1)
    return f"{name[:7]}..." if len(name) > 8 else name


def _normalize_gpu_name(name: str) -> str:
    for pattern, short in _GPU_NAME_PATTERNS:
        if pattern.search(name):
            return short
    return _shorten_gpu_name(name)


def _normalize_gpu_counts(by_type: dict | None) -> dict[str, int]:
    """Filter out zero/CPU entries and collapse raw GPU names into short SKUs."""
    out: dict[str, int] = {}
    for raw_type, count in (by_type or {}).items():
        if count <= 0 or raw_type in _CPU_MACHINE_TYPES:
            continue
        short = _normalize_gpu_name(raw_type)
        if not short:
            continue
        out[short] = out.get(short, 0) + count
    return out


def _get_metrics(client: SyncServerlessClient) -> dict:
    """Fetch the raw /applications/metrics payload."""
    rest_client = client._create_rest_client()
    with httpx.Client(
        base_url=rest_client.base_url,
        headers=rest_client.get_headers(),
        timeout=30,
        verify=not flags.TEST_MODE,
        follow_redirects=True,
    ) as http:
        resp = http.get("/applications/metrics")
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"Failed to fetch metrics: {resp.status_code} {resp.text}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected metrics response format: {resp.text}")
    return data
