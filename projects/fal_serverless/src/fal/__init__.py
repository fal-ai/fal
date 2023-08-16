from __future__ import annotations

from fal.api import FalServerlessHost, LocalHost, cached, isolated
from fal.decorators import download_file, download_weights
from fal.sdk import FalServerlessKeyCredentials
from fal.sync import sync_dir

local = LocalHost()
serverless = FalServerlessHost()

# DEPRECATED - use serverless instead
cloud = FalServerlessHost()

DBT_FAL_IMPORT_NOTICE = """
The dbt tool `fal` and `dbt-fal` adapter have been merged into a single tool.
Please import from the `fal.dbt` module instead.
Running `pip install dbt-fal` will install the new tool and the adapter alongside.
Then import from the `fal.dbt` module like

    from fal.dbt import {name}

"""

# Avoid printing on non-direct imports
def __getattr__(name: str):
    if name in (
        "NodeStatus",
        "FalDbt",
        "DbtModel",
        "DbtSource",
        "DbtTest",
        "DbtGenericTest",
        "DbtSingularTest",
        "Context",
        "CurrentModel",
    ):
        raise ImportError(DBT_FAL_IMPORT_NOTICE.format(name=name))

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
