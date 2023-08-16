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


# NOTE: taken from dbt-core: https://github.com/dbt-labs/dbt-core/blob/ac539fd5cf325cfb5315339077d03399d575f570/core/dbt/adapters/__init__.py#L1-L7
# N.B.
# This will add to the package’s __path__ all subdirectories of directories on sys.path named after the package which effectively combines both modules into a single namespace (dbt.adapters)
# The matching statement is in plugins/postgres/dbt/adapters/__init__.py

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
