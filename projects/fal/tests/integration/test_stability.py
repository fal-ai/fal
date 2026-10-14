from __future__ import annotations

import uuid
from contextlib import suppress

import pytest
from isolate.backends.common import active_python
from pydantic import __version__ as pydantic_version

import fal
from fal.api import FalSerializationError, FalServerlessError
from fal.container import ContainerImage
from fal.toolkit.file import File
from tests.utils import git_revision_short_hash

# Each test deploys and runs an isolated function inline, so the deploy
# counts against the timer and needs more than the unit default.
pytestmark = pytest.mark.timeout(180)

PACKAGE_NAME = "fal"


def test_missing_dependencies_nested_server_error(isolated_client):
    @isolated_client()
    def test1():
        return "hello"

    @isolated_client()
    def test2():
        test1()

    with pytest.raises(
        FalSerializationError,
        match="Could not find module 'fal'",
    ) as exc_info:
        test2()

    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)
    assert exc_info.value.__cause__.name == "fal"


def test_regular_function(isolated_client):
    @isolated_client("virtualenv")
    def regular_function():
        return 42

    assert regular_function() == 42

    @isolated_client("virtualenv")
    def mult(a, b):
        return a * b

    assert mult(5, 2) == 10


def test_regular_function_in_a_container(isolated_client):
    @isolated_client("container")
    def regular_function():
        return 42

    assert regular_function() == 42

    @isolated_client("container")
    def mult(a, b):
        return a * b

    assert mult(5, 2) == 10


def test_container_no_venv(isolated_client):
    actual_python = active_python()

    @isolated_client(
        "container",
        image=ContainerImage.from_dockerfile_str(
            f"""
            FROM python:{actual_python}
            # {uuid.uuid4()}
            """
        ),
    )
    def myfunc():
        return 42

    assert myfunc() == 42


def test_container_venv(isolated_client):
    actual_python = active_python()

    @isolated_client(
        "container",
        image=ContainerImage.from_dockerfile_str(
            f"""
            FROM python:{actual_python}
            # {uuid.uuid4()}
            RUN python -m venv .venv
            ENV PATH="/root/.venv/bin:$PATH"
            """
        ),
    )
    def myfunc():
        return 42

    assert myfunc() == 42


@pytest.mark.flaky(max_runs=3)
def test_regular_function_in_a_container_with_custom_image(isolated_client):
    actual_python = active_python()

    @isolated_client(
        "container",
        image=ContainerImage.from_dockerfile_str(
            f"""
            FROM python:{actual_python}-slim
            # {git_revision_short_hash()}
            RUN env
            """
        ),
    )
    def regular_function():
        return 42

    assert regular_function() == 42

    @isolated_client(
        "container",
        image=ContainerImage.from_dockerfile_str(
            f"FROM python:{actual_python}-slim\n# {git_revision_short_hash()}"
        ),
    )
    def mult(a, b):
        return a * b

    assert mult(5, 2) == 10


def test_function_pipelining(isolated_client):
    @isolated_client("virtualenv")
    def regular_function():
        return 42

    @isolated_client("virtualenv")
    def calling_function(result):
        return result * 2

    assert calling_function(regular_function()) == 84


def test_function_exception(isolated_client):
    class FooException(Exception):
        pass

    class BarException(Exception):
        pass

    class MyFuncException(Exception):
        pass

    def foo():
        raise FooException("foo")

    def bar():
        try:
            foo()
        except Exception as exc:
            raise BarException("bar") from exc

    @isolated_client()
    def myfunc():
        try:
            bar()
        except Exception as exc:
            raise MyFuncException("myfunc") from exc

    with pytest.raises(MyFuncException) as excinfo:
        myfunc()

    bar_exc = excinfo.value.__cause__
    assert isinstance(bar_exc, BarException)
    assert isinstance(bar_exc.__cause__, FooException)


@pytest.mark.skip(
    reason="See https://github.com/fal-ai/fal/issues/169",
)
def test_function_calling_other_function(isolated_client):
    try:
        import importlib.metadata as importlib_metadata
    except ImportError:
        import importlib_metadata

    package_version = importlib_metadata.version(PACKAGE_NAME)

    @isolated_client("virtualenv")
    def regular_function():
        return 42

    @isolated_client(
        "virtualenv",
        requirements=[
            f"{PACKAGE_NAME}>=0.6.19,<={package_version}",
        ],
    )
    def calling_function(recurse):
        if recurse:
            return calling_function(recurse=False)
        else:
            return regular_function()

    # One level of direct calls
    assert calling_function(recurse=False) == 42

    # Two levels of direct calls
    assert calling_function(recurse=True) == 42


@pytest.mark.skip(
    reason="Nested isolated functions require an explicit fal dependency; "
    "remove once dependency inference is supported.",
)
def test_dependency_inference(isolated_client):
    @isolated_client("virtualenv")
    def regular_function():
        return 42

    @isolated_client("virtualenv")
    def calling_function():
        return regular_function()

    assert calling_function() == 42


def test_unserializable_input_function(isolated_client):
    # When the function can not be serialized at all (has a reference to an unpicklable
    # object (like a frame)).

    @isolated_client("virtualenv")
    def unpicklable_input_function_client(default=__import__("sys")._getframe(0)):
        return default

    with pytest.raises(
        FalSerializationError, match="Error while serializing the given object"
    ) as exc_info:
        unpicklable_input_function_client()

    assert isinstance(exc_info.value.__cause__, TypeError)
    assert "frame" in str(exc_info.value.__cause__)


def test_unserializable_return(isolated_client):
    # When the return of the function can not be serialized.

    @isolated_client("virtualenv")
    def unpicklable_return():
        import sys

        return sys._getframe(0)

    with pytest.raises(FalServerlessError, match="Error while serializing"):
        unpicklable_return()


def test_missing_dependencies_on_client(isolated_client):
    module_name = f"runner_only_dependency_{uuid.uuid4().hex}"

    @isolated_client("virtualenv")
    def remote_only_module():
        import sys
        import types

        module = types.ModuleType(module_name)
        # Registration makes cloudpickle serialize this module by reference.
        # It must remain registered until the agent serializes the return value.
        sys.modules[module_name] = module
        return module

    with pytest.raises(
        FalSerializationError, match=f"Could not find module '{module_name}'"
    ) as exc_info:
        remote_only_module()

    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)
    assert exc_info.value.__cause__.name == module_name


def test_unserializable_exception(isolated_client):
    @isolated_client("virtualenv")
    def unpicklable_input_function_client():
        class T(Exception):
            frame = __import__("sys")._getframe(0)

        raise T("remote exception with an unpicklable frame")

    with pytest.raises(
        FalServerlessError, match="Error while serializing the execution result"
    ):
        unpicklable_input_function_client()


@pytest.mark.skip(
    reason="Dependency conflicts are detected, but the backend returns a generic "
    "environment build error instead of the resolver diagnostic.",
)
def test_client_superseeding_dependencies_crash(isolated_client):
    # E.g. I depend on tensorboard which depends on dill==0.3.2
    # but the agent overrides it with dill==0.3.5.1 so I get an
    # unexpected error at runtime (instead of at environment creation
    # time)

    @isolated_client("virtualenv", requirements=["dill==0.3.2"])
    def conflicting_environment():
        import dill

        assert dill.__version__ == "0.3.2"

    # This should crash at the build stage
    with pytest.raises(
        FalServerlessError, match="package versions have conflicting dependencies."
    ):
        conflicting_environment()


@pytest.mark.skip(
    reason="OOM currently surfaces as a closed runner stream, not an insufficient "
    "memory diagnostic; remove once OOM reporting is reliable.",
)
def test_memory_overflow_crash_on_run(isolated_client):
    @isolated_client("virtualenv")
    def memory_overflow_crash_on_run():
        import os

        objects = []
        while True:
            objects.append(os.urandom(1024**3))

    with pytest.raises(FalServerlessError, match="Insufficient memory"):
        memory_overflow_crash_on_run()


def test_keepalive_after_agent_exit(isolated_client):
    # Should work (fresh)
    @isolated_client("virtualenv")
    def regular_function():
        return 42

    assert regular_function() == 42

    # This uses the cached agent process but it will
    # kill it at the end.
    with suppress(BaseException):

        @isolated_client("virtualenv")
        def process_crash_without_catching():
            import os

            os._exit(0)

        process_crash_without_catching()

    # This should start a fresh agent process.
    @isolated_client("virtualenv")
    def regular_function():
        return 42

    assert regular_function() == 42


def test_faulty_setup_function(isolated_client):
    def good_setup_function():
        return 42

    def bad_setup_function():
        raise ValueError()

    @isolated_client("virtualenv")
    def regular_function(result):
        return result * 2

    first_case = regular_function.on(setup_function=good_setup_function)
    assert first_case() == 84

    second_case = regular_function.on(setup_function=bad_setup_function)
    with pytest.raises(ValueError):
        second_case()


def test_unserializable_setup_function(isolated_client):
    def unpicklable_setup_function():
        import sys

        return sys._getframe(0)

    @isolated_client("virtualenv", setup_function=unpicklable_setup_function)
    def regular_function(result):
        return str(result)

    # This works fine since the setup function's result is not returned
    # to the client (but rather just kept on the server).
    assert isinstance(regular_function(), str)


def test_big_message(isolated_client):
    # Default gRPC max message size is 4MB so let's
    # try doubling that.
    data_length = 8 * (1024**2)

    @isolated_client("virtualenv", machine_type="M")
    def big_return_function(data_length):
        return b"0" * data_length

    assert len(big_return_function(1)) == 1
    assert len(big_return_function(data_length)) == data_length


def test_futures(isolated_client):
    from concurrent.futures import wait

    @isolated_client("virtualenv")
    def regular_function(n):
        return n * 2

    future = regular_function.submit(1)
    assert future.result() == 2

    future_2 = regular_function.submit(2)
    future_3 = regular_function.submit(3)
    future_4 = regular_function.submit(4)
    wait([future_2, future_3, future_4])

    # Ensure the futures are done in the correct order
    # (no mix and match between arguments and the returned
    # future).
    assert future_2.result() == 4
    assert future_3.result() == 6
    assert future_4.result() == 8


@pytest.mark.skip(
    reason="Successive inline calls can start different runners despite keep_alive; "
    "remove once this test can reliably reuse a runner for its in-memory cache.",
)
def test_cached_function(isolated_client, capsys, monkeypatch):
    import inspect
    import time

    test_stamp = time.time()
    real_getsource = inspect.getsource

    # For ensuring between test runs the cached functions
    # are gone, we add a timestamp to the source code.
    def add_timestamp_to_source(func):
        return real_getsource(func) + f"\n# {test_stamp}"

    monkeypatch.setattr(inspect, "getsource", add_timestamp_to_source)

    @fal.cached
    def get_pipe():
        import time

        print("computing")
        time.sleep(1)  # slow IO
        return "pipe"

    @fal.cached
    def factorial(n: int) -> int:
        import math
        import time

        print("computing")
        time.sleep(1)  # slow CPU
        return math.factorial(n)

    # make sure this machine is not shared with others by using a unique requirements
    @isolated_client(
        "virtualenv",
        keep_alive=30,
        requirements=["pyjokes     "],
    )
    def regular_function(n):
        if get_pipe() == "pipe":
            return factorial(n)

    assert regular_function(4) == 24
    out, err = capsys.readouterr()
    assert out.count("computing") == 2

    # pipe is now cached, but using a different factorial(n)
    assert regular_function(3) == 6
    out, err = capsys.readouterr()
    assert out.count("computing") == 1

    # This should be all cached
    assert regular_function(4) == 24
    out, err = capsys.readouterr()
    assert out.count("computing") == 0

    # So does thiss
    assert regular_function(3) == 6
    out, err = capsys.readouterr()
    assert out.count("computing") == 0

    # But this should not be cached
    assert regular_function(5) == 120
    out, err = capsys.readouterr()
    assert out.count("computing") == 1


def test_pydantic_serialization(isolated_client):
    from pydantic import BaseModel, Field

    class MathQuery(BaseModel):
        x: int = Field(gt=0, description="The first operand")
        y: int = Field(gt=0, description="The second operand")

    class MathResult(BaseModel):
        result: int = Field(description="The result of the operation")

    @isolated_client("virtualenv", requirements=[f"pydantic=={pydantic_version}"])
    def add(query: MathQuery) -> MathResult:
        return MathResult(result=query.x + query.y)

    result = add(MathQuery(x=1, y=2))
    assert result.result == 3
    assert result == MathResult(result=3)


def test_serve_on_off(isolated_client):
    from pydantic import BaseModel, Field

    class MathQuery(BaseModel):
        x: int = Field(gt=0, description="The first operand")
        y: int = Field(gt=0, description="The second operand")

    class MathResult(BaseModel):
        result: int = Field(description="The result of the operation")

    @isolated_client(
        "virtualenv", serve=True, requirements=[f"pydantic=={pydantic_version}"]
    )
    def add(query: MathQuery) -> MathResult:
        return MathResult(result=query.x + query.y)

    local_first_add = add.on(serve=False)
    assert local_first_add(MathQuery(x=1, y=2)) == MathResult(result=3)

    local_first_add_2 = local_first_add.on(serve=True).on(serve=False)
    assert local_first_add_2(MathQuery(x=1, y=2)) == MathResult(result=3)


def test_worker_env_vars(isolated_client):
    from fal.flags import GRPC_HOST

    @isolated_client("virtualenv", keep_alive=5)
    def get_env_vars() -> str | None:
        import os

        return {
            "FAL_HOST": os.getenv("FAL_HOST", None),
            "FAL_KEY_ID": os.getenv("FAL_KEY_ID", None),
            "FAL_KEY_SECRET": os.getenv("FAL_KEY_SECRET", None),
        }

    env_vars = get_env_vars()
    fal_host = env_vars["FAL_HOST"]
    fal_key_id = env_vars["FAL_KEY_ID"]
    fal_key_secret = env_vars["FAL_KEY_SECRET"]

    assert fal_host, "FAL_HOST is not set"
    assert fal_host == GRPC_HOST, "FAL_HOST is not set to the correct value"
    print(f"FAL_HOST: {fal_host}, GRPC_HOST: {GRPC_HOST}")
    assert fal_key_id, "FAL_KEY_ID is not set"
    assert fal_key_secret, "FAL_KEY_SECRET is not set"


@pytest.mark.flaky(max_runs=3)
def test_fal_storage_v3(isolated_client):
    url_prefixes = ["https://v3.fal.media/files", "https://v3b.fal.media/files"]

    # v3b validates content against the claimed type; keep this test out of the
    # rate-limited unrecognized-binary path.
    file = File.from_bytes(
        b"Hello fal storage from local",
        content_type="text/plain",
        file_name="fal-storage-v3-local.txt",
        repository="fal_v3",
        fallback_repository=None,
    )
    assert any(file.url.startswith(url_prefix) for url_prefix in url_prefixes)
    assert file.as_bytes().decode().endswith("local")

    @isolated_client(
        serve=True, requirements=[f"pydantic=={pydantic_version}", "tomli"]
    )
    def hello_file():
        # Run in the isolated environment
        return File.from_bytes(
            b"Hello fal storage from isolated",
            content_type="text/plain",
            file_name="fal-storage-v3-isolated.txt",
            repository="fal_v3",
            save_kwargs={
                "multipart": False,
                "multipart_threshold": 1024 * 1024,
                "multipart_chunk_size": 1024 * 1024,
                "multipart_max_concurrency": 10,
                "object_lifecycle_preference": {},
            },
            fallback_repository=None,
        )

    local_fn = hello_file.on(serve=False)
    file = local_fn()
    assert any(file.url.startswith(url_prefix) for url_prefix in url_prefixes)
    assert file.as_bytes().decode().endswith("isolated")


def test_pydantic_file_model_fields_match_in_isolation(isolated_client):
    from pydantic import BaseModel

    class RemoteOutput(BaseModel):
        video: File

    @isolated_client(
        "virtualenv", requirements=[f"pydantic=={pydantic_version}", "pytest"]
    )
    def check_models():
        out = RemoteOutput(
            video=File(url="https://example.com/a"),
        )

        out_dict = out.dict() if pydantic_version.startswith("1.") else out.model_dump()
        assert out_dict == {
            "video": {
                "url": "https://example.com/a",
                "content_type": None,
                "file_name": None,
                "file_size": None,
            },
        }

    check_models()
