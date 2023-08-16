from __future__ import annotations

from fal import isolated
from fal.api import FalServerlessHost

host = FalServerlessHost(url="localhost:6005")


@isolated("virtualenv", requirements=["pyjokes==0.5.0"], host=host)
def get_pyjokes_version():
    import pyjokes

    return pyjokes.__version__


assert get_pyjokes_version() == "0.5.0"
