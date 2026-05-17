from __future__ import annotations

import os
import subprocess
import sys

from fal.console.icons import _select_icon


def test_select_icon_uses_ascii_fallback():
    assert _select_icon("unicode", "ascii", ascii_only=True) == "ascii"


def test_select_icon_uses_unicode_when_supported():
    assert _select_icon("unicode", "ascii", ascii_only=False) == "unicode"


def test_cross_icon_renders_with_cp1252_output():
    script = "\n".join(
        [
            "from fal.console import console",
            "from fal.console.icons import CROSS_ICON, SECTION_ICON",
            'console.print(f"{CROSS_ICON} bad")',
            'console.print(f"{SECTION_ICON} section")',
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
    assert b"x bad" in result.stdout
    assert b"> section" in result.stdout
