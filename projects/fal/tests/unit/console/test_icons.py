from __future__ import annotations

import io
import os
import subprocess
import sys
import tokenize
from pathlib import Path

from fal.console.icons import _select_icon

PROBLEMATIC_CONSOLE_GLYPHS = {
    "\u2022": "BULLET_ICON",
    "\u25b8": "SECTION_ICON",
    "\u2713": "CHECK_ICON",
    "\u2718": "CROSS_ICON",
    "\u23f3": "STATUS_QUEUED_ICON",
    "\U0001f504": "STATUS_PROGRESS_ICON",
    "\u2705": "STATUS_DONE_ICON",
    "\U0001f927": "WORKFLOW_LOADED_ICON",
    "\U0001f389": "WORKFLOW_COMPLETE_ICON",
}


def test_select_icon_uses_ascii_fallback():
    assert _select_icon("unicode", "ascii", ascii_only=True) == "ascii"


def test_select_icon_uses_unicode_when_supported():
    assert _select_icon("unicode", "ascii", ascii_only=False) == "unicode"


def test_cross_icon_renders_with_cp1252_output():
    script = "\n".join(
        [
            "from fal.console import console",
            "from fal.console.icons import (",
            "    BULLET_ICON,",
            "    CROSS_ICON,",
            "    SECTION_ICON,",
            "    STATUS_DONE_ICON,",
            "    STATUS_PROGRESS_ICON,",
            "    STATUS_QUEUED_ICON,",
            "    WORKFLOW_COMPLETE_ICON,",
            "    WORKFLOW_LOADED_ICON,",
            ")",
            'console.print(f"{BULLET_ICON} item")',
            'console.print(f"{CROSS_ICON} bad")',
            'console.print(f"{SECTION_ICON} section")',
            'console.print(f"{STATUS_QUEUED_ICON} queued")',
            'console.print(f"{STATUS_PROGRESS_ICON} progress")',
            'console.print(f"{STATUS_DONE_ICON} done")',
            'console.print(f"{WORKFLOW_LOADED_ICON} loaded")',
            'console.print(f"{WORKFLOW_COMPLETE_ICON} complete")',
        ]
    )
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env=env,
        text=False,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"- item" in result.stdout
    assert b"x bad" in result.stdout
    assert b"> section" in result.stdout
    assert b"... queued" in result.stdout
    assert b"~ progress" in result.stdout
    assert b"+ done" in result.stdout
    assert b"* loaded" in result.stdout
    assert b"+ complete" in result.stdout


def test_console_output_uses_icon_fallbacks_for_problematic_glyphs():
    source_root = Path(__file__).resolve().parents[3] / "src" / "fal"
    allowed_files = {Path("console/icons.py")}
    offenders = []

    for path in sorted(source_root.rglob("*.py")):
        rel_path = path.relative_to(source_root)
        if rel_path in allowed_files:
            continue

        text = path.read_text(encoding="utf-8")
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            if token.type != tokenize.STRING:
                continue
            for glyph, fallback_name in PROBLEMATIC_CONSOLE_GLYPHS.items():
                if glyph in token.string:
                    offenders.append(f"{rel_path}:{token.start[0]} use {fallback_name}")

    assert not offenders, "\n".join(offenders)
