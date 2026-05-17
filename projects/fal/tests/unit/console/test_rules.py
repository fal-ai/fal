from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import MagicMock

from rich.rule import Rule

from fal.console.rules import print_rule


def test_print_rule_uses_rich_rule_unless_ascii_only_is_true():
    console = MagicMock()

    print_rule(console, "Title", style="dim")

    rule = console.print.call_args.args[0]
    assert isinstance(rule, Rule)


def test_print_rule_renders_with_cp1252_output():
    script = "\n".join(
        [
            "from rich.text import Text",
            "from fal.console import console",
            "from fal.console.rules import print_rule",
            'print_rule(console, Text("Title"), style="dim")',
            'print_rule(console, Text("app[v2]"), style="dim")',
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
    assert b"app[v2]" in result.stdout
    assert b"-" in result.stdout
