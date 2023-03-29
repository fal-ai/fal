from __future__ import annotations

import os


def bool_envvar(name: str):
    if name in os.environ:
        val = os.environ[name].lower().strip()
        return val != "0" and val != "false" and val != ""
    return False


TEST_MODE = bool_envvar("ISOLATE_TEST_MODE")
