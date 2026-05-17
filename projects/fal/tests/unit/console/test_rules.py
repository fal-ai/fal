from __future__ import annotations

import os
import subprocess
import sys


def test_print_rule_renders_with_cp1252_output():
    script = "\n".join(
        [
            "from rich.text import Text",
            "from fal.console import console",
            "from fal.console.rules import print_rule",
            'print_rule(console, Text("Title"), style="dim")',
            'print_rule(console, style="dim")',
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
    assert b"Title" in result.stdout
    assert b"-" in result.stdout
