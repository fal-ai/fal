from unittest.mock import MagicMock, patch

import pytest

from fal.api._metrics import (
    _normalize_gpu_counts,
    _normalize_gpu_name,
    _shorten_gpu_name,
)
from fal.api.apps import app_gpus


@pytest.mark.parametrize(
    "raw, expected",
    [
        # SKU rules collapse driver variants
        ("NVIDIA H100 80GB HBM3", "H100"),
        ("NVIDIA H100 80GB HBM3 MIG 3g.40gb", "H100"),
        ("NVIDIA H200", "H200"),
        ("NVIDIA B200", "B200"),
        ("NVIDIA A100-SXM4-80GB", "A100"),
        ("NVIDIA A100-SXM4-40GB", "A100"),
        ("NVIDIA A100 PCIE 40GB", "A100"),
        ("NVIDIA L40", "L40"),
        ("NVIDIA L40S", "L40"),
        ("NVIDIA RTX 5090", "5090"),
        ("RTX-5090", "5090"),
        ("GPU-H100", "H100"),
        # Fallthrough to shortenGpuName
        ("NVIDIA RTX A6000", "RTX A6000"),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "RTX 6000"),
    ],
)
def test_normalize_gpu_name(raw, expected):
    assert _normalize_gpu_name(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        # CPU plans return empty so they get filtered downstream
        ("M", ""),
        ("L", ""),
        ("XS", ""),
        # RTX matcher with optional PRO
        ("NVIDIA RTX A6000", "RTX A6000"),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "RTX 6000"),
        # MIG matcher
        ("NVIDIA H100 MIG 3g.40gb", "H100 MIG"),
        # AMD MI series
        ("AMD MI300X", "MI300X"),
        # Generic model fallback (requires 2+ digits after the letter)
        ("Tesla V100", "V100"),
        # Truncation fallback for unmatched names longer than 8 chars
        ("Unknown123Foo", "Unknown..."),
    ],
)
def test_shorten_gpu_name(raw, expected):
    assert _shorten_gpu_name(raw) == expected


def test_normalize_gpu_counts_collapses_variants_and_drops_noise():
    raw = {
        "NVIDIA H100 80GB HBM3": 400,
        "NVIDIA H100 80GB HBM3 MIG 3g.40gb": 30,  # collapses into H100
        "NVIDIA H200": 280,
        "NVIDIA A100-SXM4-80GB": 5,
        "NVIDIA A100-SXM4-40GB": 3,  # collapses into A100
        "M": 0,  # filtered (CPU + zero)
        "L": 0,
        "S": 0,
        "XS": 0,
        "XL": 0,
    }
    result = _normalize_gpu_counts(raw)
    assert result == {"H100": 430, "H200": 280, "A100": 8}


def test_normalize_gpu_counts_filters_cpu_types_with_nonzero_counts():
    # Defensive: even if the endpoint accidentally reports a non-zero CPU
    # plan in the GPU breakdown, we still filter it out.
    raw = {"NVIDIA H100": 10, "M": 5, "L": 3}
    assert _normalize_gpu_counts(raw) == {"H100": 10}


def test_normalize_gpu_counts_handles_none_and_empty():
    assert _normalize_gpu_counts(None) == {}
    assert _normalize_gpu_counts({}) == {}


@patch("fal.api.apps._get_metrics")
def test_app_gpus_normalizes_per_app(mock_get_metrics):
    mock_get_metrics.return_value = {
        "apps": {
            "fal-ai/flux": {
                "runner_count": 3,
                "gpu_count": 11,
                "gpu_count_by_type": {
                    "NVIDIA H100 80GB HBM3": 8,
                    "NVIDIA H100 80GB HBM3 MIG 3g.40gb": 3,
                    "M": 0,
                },
                "cpu_count": 0,
                "cpu_count_by_type": {},
                "queue_size": 0,
            },
        },
        "summary": {},
    }
    result = app_gpus(MagicMock(), "fal-ai/flux")
    assert result == {"gpus": {"H100": 11}, "total": 11}


@patch("fal.api.apps._get_metrics")
def test_app_gpus_raises_when_app_missing(mock_get_metrics):
    mock_get_metrics.return_value = {"apps": {}, "summary": {}}
    with pytest.raises(RuntimeError, match="not found in metrics"):
        app_gpus(MagicMock(), "fal-ai/missing")


_FLUX_APP = {
    "runner_count": 1,
    "gpu_count": 4,
    "gpu_count_by_type": {"NVIDIA H100 80GB HBM3": 4},
    "cpu_count": 0,
    "cpu_count_by_type": {},
    "queue_size": 0,
}


@patch("fal.api.apps._get_metrics")
def test_app_gpus_accepts_bare_name(mock_get_metrics):
    mock_get_metrics.return_value = {
        "apps": {"fal-ai/flux": _FLUX_APP},
        "summary": {},
    }
    # Bare name resolves to the namespaced key
    assert app_gpus(MagicMock(), "flux") == {"gpus": {"H100": 4}, "total": 4}


@patch("fal.api.apps._get_metrics")
def test_app_gpus_accepts_namespaced_name(mock_get_metrics):
    mock_get_metrics.return_value = {
        "apps": {"fal-ai/flux": _FLUX_APP},
        "summary": {},
    }
    assert app_gpus(MagicMock(), "fal-ai/flux") == {"gpus": {"H100": 4}, "total": 4}


@patch("fal.api.apps._get_metrics")
def test_app_gpus_ambiguous_bare_name(mock_get_metrics):
    mock_get_metrics.return_value = {
        "apps": {
            "team-a/flux": _FLUX_APP,
            "team-b/flux": _FLUX_APP,
        },
        "summary": {},
    }
    with pytest.raises(RuntimeError, match="ambiguous"):
        app_gpus(MagicMock(), "flux")
