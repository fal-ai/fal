from __future__ import annotations

from fal_serverless import isolated
from fal_serverless.api import KoldstartHost

host = KoldstartHost(url="localhost:6005")


@isolated("virtualenv", requirements=["pyjokes==0.5.0"], host=host)
def get_pyjokes_version():
    import pyjokes

    return pyjokes.__version__


assert get_pyjokes_version() == "0.5.0"
