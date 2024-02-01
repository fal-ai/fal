import time
from pathlib import Path

import fal
import pytest
from fal import apps
from pydantic import BaseModel
from pydantic import __version__ as PYDANTIC_VERSION

# This is just a rough first draft of how we would check the Pydantic version
IS_PYDANTIC_2 = PYDANTIC_VERSION[0] == "2"

if IS_PYDANTIC_2:
    import pydantic_patch  # noqa


class Input(BaseModel):
    lhs: int
    rhs: int
    wait_time: int = 0


class Output(BaseModel):
    result: int


pydantic_patch_contents = Path("pydantic_patch.py").read_text()


def apply_patch():
    """Load the patch via a package in a temporary directory prepended to the PATH."""
    import sys
    from tempfile import mkdtemp

    package_dir = mkdtemp()

    path = Path(package_dir) / "pydantic_patch.py"
    path.write_text(pydantic_patch_contents)

    sys.path.insert(0, package_dir)


@fal.function(
    keep_alive=0,
    machine_type="S",
    serve=True,
    max_concurrency=1,
    setup_function=apply_patch,
)
def addition_app(input: Input) -> Output:
    print("starting...")
    for _ in range(input.wait_time):
        print("sleeping...")
        time.sleep(1)

    return Output(result=input.lhs + input.rhs)


@pytest.fixture(scope="module")
def temp_app():
    """Create a temporary app, register it, and return the ID of it."""
    from fal.cli import _get_user_id

    app_revision = addition_app.host.register(
        func=addition_app.func, options=addition_app.options
    )
    user_id = _get_user_id()
    yield f"{user_id}-{app_revision}"


def test_app_client(temp_app: str):
    response = apps.run(test_app, arguments={"lhs": 1, "rhs": 2})
    assert response["result"] == 3

    response = apps.run(test_app, arguments={"lhs": 2, "rhs": 3, "wait_time": 1})
    assert response["result"] == 5
