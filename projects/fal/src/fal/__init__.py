from __future__ import annotations

from fal import apps

# TODO: DEPRECATED - use function instead
from fal.api import FalServerlessHost, LocalHost, cached
from fal.api import function
from fal.api import function as isolated
from fal.app import App, endpoint, realtime, wrap_app
from fal.sdk import FalServerlessKeyCredentials
from fal.sync import sync_dir

local = LocalHost()
serverless = FalServerlessHost()

# DEPRECATED - use serverless instead
cloud = FalServerlessHost()


# NOTE: This makes `import fal.dbt` import the `dbt-fal` module and `import fal` import the `fal` module
# NOTE: taken from dbt-core: https://github.com/dbt-labs/dbt-core/blob/ac539fd5cf325cfb5315339077d03399d575f570/core/dbt/adapters/__init__.py#L1-L7
# N.B.
# This will add to the package’s __path__ all subdirectories of directories on sys.path named after the package which effectively combines both modules into a single namespace (dbt.adapters)
# The matching statement is in plugins/postgres/dbt/adapters/__init__.py

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
