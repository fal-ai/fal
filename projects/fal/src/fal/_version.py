import json
import os
import tempfile
from typing import Any, Dict, Optional

try:
    from ._fal_version import version as __version__  # type: ignore[import]
    from ._fal_version import version_tuple  # type: ignore[import]
except ImportError:
    __version__ = "UNKNOWN"
    version_tuple = (0, 0, __version__)  # type: ignore[assignment]


_PYPI_URL = "https://pypi.org/pypi/fal/json"
_PYPI_CACHE_TTL = 60 * 60  # 1 hour
_PYPI_CACHE_PATH = os.path.expanduser("~/.fal/cache/pypi.json")
_URLOPEN_TIMEOUT = 1


def _write_pypi_cache(data: Dict[str, Any]) -> None:
    cache_dir = os.path.dirname(_PYPI_CACHE_PATH)
    os.makedirs(cache_dir, exist_ok=True)
    prefix = os.path.basename(_PYPI_CACHE_PATH) + ".tmp."
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=cache_dir,
        prefix=prefix,
        delete=False,
    ) as fobj:
        fobj.write(json.dumps(data))
        os.rename(fobj.name, _PYPI_CACHE_PATH)


def _get_pypi_cache() -> Optional[Dict[str, Any]]:
    import time

    try:
        mtime = os.path.getmtime(_PYPI_CACHE_PATH)
    except FileNotFoundError:
        return None

    if mtime + _PYPI_CACHE_TTL < time.time():
        return None

    with open(_PYPI_CACHE_PATH) as fobj:
        try:
            return json.load(fobj)
        except ValueError:
            return None


def _fetch_pypi_data() -> Dict[str, Any]:
    from urllib.request import urlopen

    response = urlopen(_PYPI_URL, timeout=_URLOPEN_TIMEOUT)
    if response.status != 200:
        raise Exception(f"Failed to fetch {_PYPI_URL}")

    data = response.read()
    return json.loads(data)


def get_latest_version() -> str:
    from fal.logging import get_logger

    logger = get_logger(__name__)

    try:
        data = _get_pypi_cache()
    except Exception:
        logger.warning("Failed to get pypi cache", exc_info=True)
        data = None

    if data is None:
        try:
            data = _fetch_pypi_data()
        except Exception:
            logger.warning("Failed to get latest fal version", exc_info=True)
            data = {}

        try:
            _write_pypi_cache(data)
        except Exception:
            logger.warning("Failed to write pypi cache", exc_info=True)

    try:
        return data["info"]["version"]
    except KeyError:
        return "0.0.0"
