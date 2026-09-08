"""Tests for fal.sync ignore handling."""

import warnings

import pytest

from fal.sync import _build_ignore_spec

IGNORE_PATTERNS = [
    "# comment",
    "",
    "*.pyc",
    "!keep.pyc",
    "node_modules/",
    "/env",
    "build/**",
]


@pytest.mark.parametrize(
    "path, expected",
    [
        ("src/main.py", False),
        ("x/y.pyc", True),
        ("keep.pyc", False),
        ("node_modules/pkg/index.js", True),
        ("env", True),
        ("sub/env", False),
        ("build/out/app.bin", True),
    ],
)
def test_ignore_spec_matches_gitignore_semantics(path, expected):
    spec = _build_ignore_spec(IGNORE_PATTERNS)
    assert spec.match_file(path) is expected


def test_empty_gitignore_ignores_nothing():
    spec = _build_ignore_spec([])
    assert not spec.match_file("anything.py")


def test_building_and_matching_emits_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = _build_ignore_spec(IGNORE_PATTERNS)
        assert spec.match_file("x/y.pyc")
