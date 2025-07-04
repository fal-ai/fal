import asyncio
import json
import secrets
import subprocess
import time
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta, timezone
from typing import Generator, List, Tuple, Union

import httpx
import pytest
from fastapi import Request, WebSocket
from httpx import HTTPStatusError
from isolate.backends.common import active_python
from openapi_fal_rest.api.applications import app_metadata
from pydantic import BaseModel
from pydantic import __version__ as pydantic_version

import fal
import fal.api as api
from fal import apps
from fal.app import AppClient, AppClientError, wrap_app
from fal.cli.deploy import User, _get_user
from fal.container import ContainerImage
from fal.exceptions import (
    AppException,
    FalServerlessException,
    FieldException,
    RequestCancelledException,
)
from fal.exceptions._cuda import _CUDA_OOM_MESSAGE, _CUDA_OOM_STATUS_CODE
from fal.rest_client import REST_CLIENT
from fal.toolkit.utils.endpoint import cancel_on_disconnect
from fal.workflows import Workflow


class Input(BaseModel):
    lhs: int
    rhs: int
    wait_time: int = 0


class StatefulInput(BaseModel):
    value: int


class Output(BaseModel):
    result: int


actual_python = active_python()


def git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


@fal.function(
    keep_alive=60,
    machine_type=["S", "M"],
    serve=True,
    max_concurrency=1,
    requirements=[f"pydantic=={pydantic_version}"],
)
def addition_app(input: Input) -> Output:
    print("starting...")
    for _ in range(input.wait_time):
        print("sleeping...")
        time.sleep(1)

    return Output(result=input.lhs + input.rhs)


nomad_addition_app = addition_app.on(_scheduler="nomad")


@fal.function(
    kind="container",
    image=ContainerImage.from_dockerfile_str(
        f"FROM python:{actual_python}-slim\n# {git_revision_short_hash()}",
    ),
    keep_alive=60,
    machine_type="S",
    serve=True,
    max_concurrency=1,
)
def container_addition_app(input: Input) -> Output:
    print("starting...")
    for _ in range(input.wait_time):
        print("sleeping...")
        time.sleep(1)

    return Output(result=input.lhs + input.rhs)


@fal.function(
    kind="container",
    image=ContainerImage.from_dockerfile_str(
        f"""FROM python:{actual_python}-slim\n# {git_revision_short_hash()}
ARG OUTPUT="built incorrectly"
ENV OUTPUT="$OUTPUT"
""",
        build_args={"OUTPUT": "built with build args"},
    ),
    keep_alive=60,
    machine_type="S",
    serve=True,
    max_concurrency=1,
)
def container_build_args_app() -> str:
    import os

    return os.environ["OUTPUT"]


@fal.function(
    keep_alive=300,
    requirements=["fastapi", "uvicorn", "pydantic==1.10.18"],
    machine_type="S",
    max_concurrency=1,
    max_multiplexing=30,
    exposed_port=8000,
)
def calculator_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from uvicorn import run

    app = FastAPI()

    def _wait(wait_time: int):
        print("starting...")
        for _ in range(wait_time):
            print("sleeping...")
            time.sleep(1)

    @app.post("/add")
    def add(input: Input) -> Output:
        _wait(input.wait_time)
        return Output(result=input.lhs + input.rhs)

    @app.post("/subtract")
    def subtract(input: Input) -> Output:
        _wait(input.wait_time)
        return Output(result=input.lhs - input.rhs)

    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_headers=("*"),
        allow_methods=("*"),
        allow_origins=("*"),
    )

    run(app, host="0.0.0.0", port=8080)


class StatefulAdditionApp(fal.App, keep_alive=300, max_concurrency=1):
    machine_type = "S"

    async def setup(self):
        self.counter = 0

    @fal.endpoint("/reset")
    def reset(self) -> Output:
        self.counter = 0
        return Output(result=self.counter)

    @fal.endpoint("/increment")
    def increment(self, input: StatefulInput) -> Output:
        self.counter += input.value
        return Output(result=self.counter)

    @fal.endpoint("/decrement")
    def decrement(self, input: StatefulInput) -> Output:
        self.counter -= input.value
        return Output(result=self.counter)


class SleepInput(BaseModel):
    wait_time: int


class SleepOutput(BaseModel):
    slept: bool = True


class SleepApp(fal.App, keep_alive=300, max_concurrency=1):
    machine_type = "XS"

    @fal.endpoint("/")
    async def sleep(self, input: SleepInput) -> SleepOutput:
        for _ in range(input.wait_time):
            print("sleeping...", flush=True)
            await asyncio.sleep(1)
        return SleepOutput(slept=True)


class ExceptionApp(fal.App, keep_alive=300, max_concurrency=1):
    machine_type = "XS"

    @fal.endpoint("/fail")
    def fail(self) -> Output:
        raise Exception("this app is designed to fail!")

    @fal.endpoint("/app-exception")
    def app_exception(self) -> Output:
        raise AppException(message="this app is designed to fail", status_code=401)

    # While making the request provide payload as {"lhs": 1, "rhs": 2}
    @fal.endpoint("/field-exception")
    def field_exception(self, input: Input) -> Output:
        raise FieldException(
            field="rhs",
            message="rhs must be an integer",
        )

    @fal.endpoint("/cuda-exception")
    def cuda_exception(self) -> Output:
        # mimicking error message from PyTorch (https://github.com/pytorch/pytorch/blob/6c65fd03942415b68040e102c44cf5109d2d851e/c10/cuda/CUDACachingAllocator.cpp#L1234C12-L1234C30)
        raise RuntimeError("CUDA out of memory")

    @fal.endpoint("/cuda-exception-2")
    def cuda_exception_2(self) -> Output:
        # https://github.com/pytorch/pytorch/issues/112377
        raise RuntimeError("NVML_SUCCESS == r INTERNAL ASSERT FAILED")

    @fal.endpoint("/cuda-exception-3")
    def cuda_exception_3(self) -> Output:
        raise RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR")


class CancellableApp(fal.App, keep_alive=300, max_concurrency=1, request_timeout=4):
    task = None
    running = 0

    async def _sleep(self, input: Input):
        if self.running > 0:
            raise Exception("App is already running")

        self.task = asyncio.create_task(asyncio.sleep(input.wait_time))
        try:
            self.running += 1
            await self.task
        except asyncio.CancelledError:
            print("Task was cancelled")
            if not self.task.done():
                self.task.cancel()
                with suppress(Exception):
                    await self.task

            raise RequestCancelledException("Request cancelled by the client.")
        finally:
            self.task = None
            self.running -= 1
        return Output(result=input.lhs + input.rhs)

    @fal.endpoint("/")
    async def sleep(self, input: Input) -> Output:
        return await self._sleep(input)

    @fal.endpoint("/well-handled")
    async def well_handled(self, input: Input, request: Request) -> Output:
        async with cancel_on_disconnect(request):
            return await self._sleep(input)

    @fal.endpoint("/cancel")
    async def cancel_handler(self) -> Output:
        if self.task:
            self.task.cancel()
            with suppress(BaseException):
                await self.task
            self.task = None

        return Output(result=0)


class RTInput(BaseModel):
    prompt: str

    def can_batch(
        self,
        other: "RTInput",
        current_batch_size: int = 1,
    ) -> bool:
        return "don't batch" not in other.prompt


class RTOutput(BaseModel):
    text: str


class RTOutputs(BaseModel):
    texts: List[str]


class RealtimeApp(fal.App, keep_alive=300, max_concurrency=1):
    machine_type = "S"

    @fal.endpoint("/")
    def generate(self, input: RTInput) -> RTOutput:
        return RTOutput(text=input.prompt)

    @fal.endpoint("/ws", is_websocket=True)
    async def generate_ws(self, websocket: WebSocket) -> None:
        await websocket.accept()
        for _ in range(3):
            await websocket.send_json({"message": "Hello world!"})
        await websocket.close()

    @fal.realtime("/realtime")
    def generate_rt(self, input: RTInput) -> RTOutput:
        return RTOutput(text=input.prompt)

    @fal.realtime("/realtime/batched", buffering=10, max_batch_size=4)
    def generate_rt_batched(self, input: RTInput, *inputs: RTInput) -> RTOutputs:
        time.sleep(2)  # fixed cost
        return RTOutputs(texts=[input.prompt] + [i.prompt for i in inputs])


class BrokenApp(fal.App, keep_alive=300, max_concurrency=1):
    machine_type = "S"

    @fal.endpoint("/")
    def broken(self) -> Exception:
        raise Exception("this app is designed to fail")


@pytest.fixture(scope="module")
def host() -> Generator[api.Host, None, None]:
    yield addition_app.host


@pytest.fixture(scope="module")
def user() -> Generator[User, None, None]:
    user = _get_user()
    yield user


@contextmanager
def register_app(
    host: api.FalServerlessHost,
    app: Union[api.ServedIsolatedFunction, api.IsolatedFunction],
    suffix: str = "",
):
    app_alias = str(uuid.uuid4()) + "-test-alias" + ("-" + suffix if suffix else "")
    app_revision = host.register(
        func=app.func,
        options=app.options,
        application_name=app_alias,
        application_auth_mode="private",
    )
    try:
        yield app_alias, app_revision
    finally:
        with host._connection as client:
            client.delete_alias(app_alias)


@pytest.fixture(scope="module")
def base_app(host: api.FalServerlessHost):
    # running apps without aliases is no longer supported
    # so we need to create an alias for the app
    with register_app(host, addition_app, "base") as (app_alias, app_revision):
        yield app_alias, app_revision


@pytest.fixture(scope="module")
def test_app(host: api.FalServerlessHost, user: User):
    with register_app(host, addition_app, "addition") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture(scope="module")
def test_nomad_app(host: api.FalServerlessHost, user: User):
    with register_app(host, nomad_addition_app, "nomad") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture(scope="module")
def test_container_app(host: api.FalServerlessHost, user: User):
    with register_app(host, container_addition_app, "container") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture(scope="module")
def test_container_build_args_app(host: api.FalServerlessHost, user: User):
    with register_app(host, container_build_args_app, "build-args") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture(scope="module")
def test_fastapi_app(host: api.FalServerlessHost, user: User):
    with register_app(host, calculator_app, "fastapi") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture(scope="module")
def test_stateful_app(host: api.FalServerlessHost, user: User):
    stateful_app = wrap_app(StatefulAdditionApp)
    with register_app(host, stateful_app, "stateful") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture(scope="module")
def test_pydantic_validation_error():
    with AppClient.connect(StatefulAdditionApp) as client:
        yield client


@pytest.fixture(scope="module")
def test_cancellable_app(host: api.FalServerlessHost, user: User):
    cancellable_app = wrap_app(CancellableApp)
    with register_app(host, cancellable_app, "cancellable") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture(scope="module")
def test_exception_app():
    with AppClient.connect(ExceptionApp) as client:
        yield client


@pytest.fixture(scope="module")
def test_sleep_app(host: api.FalServerlessHost, user: User):
    sleep_app = wrap_app(SleepApp)
    with register_app(host, sleep_app, "sleep") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture(scope="module")
def test_realtime_app(host: api.FalServerlessHost, user: User):
    realtime_app = wrap_app(RealtimeApp)
    with register_app(host, realtime_app, "realtime") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


def test_broken_app_failure(host: api.FalServerlessHost, user: User):
    with pytest.raises(FalServerlessException) as e:
        wrap_app(BrokenApp)

    assert "Failed to generate OpenAPI" in str(e)


def test_app_client(test_app: str, test_nomad_app: str):
    response = apps.run(test_app, arguments={"lhs": 1, "rhs": 2})
    assert response["result"] == 3

    response = apps.run(test_app, arguments={"lhs": 2, "rhs": 3, "wait_time": 1})
    assert response["result"] == 5

    response = apps.run(test_nomad_app, arguments={"lhs": 1, "rhs": 2})
    assert response["result"] == 3

    response = apps.run(test_nomad_app, arguments={"lhs": 2, "rhs": 3, "wait_time": 1})
    assert response["result"] == 5


def test_ws_client(test_app: str):
    with apps.ws(test_app) as connection:
        for i in range(3):
            response = json.loads(connection.run({"lhs": 1, "rhs": i}))
            assert response["result"] == 1 + i

        for i in range(3):
            connection.send({"lhs": 2, "rhs": i})

        for i in range(3):
            # they should be in order
            response = json.loads(connection.recv())
            assert response["result"] == 2 + i


def test_app_client_old_format(base_app: Tuple[str, str], user: User):
    app_alias, _ = base_app
    old_format = f"{user.user_id}-{app_alias}"
    response = apps.run(old_format, arguments={"lhs": 1, "rhs": 2})
    assert response["result"] == 3


def test_app_client_path_included_in_app_id(test_stateful_app: str):
    response = apps.run(test_stateful_app + "/reset", arguments={})
    assert response["result"] == 0

    response = apps.run(test_stateful_app + "/increment", arguments={"value": 3})
    assert response["result"] == 3

    # if put in path we do not need to prefix with /
    response = apps.run(test_stateful_app, arguments={"value": 3}, path="increment")
    assert response["result"] == 6


def test_stateful_app_client(test_stateful_app: str):
    response = apps.run(test_stateful_app, arguments={}, path="/reset")
    assert response["result"] == 0

    response = apps.run(test_stateful_app, arguments={"value": 1}, path="/increment")
    assert response["result"] == 1

    response = apps.run(test_stateful_app, arguments={"value": 2}, path="/increment")
    assert response["result"] == 3

    response = apps.run(test_stateful_app, arguments={"value": 1}, path="/decrement")
    assert response["result"] == 2

    response = apps.run(test_stateful_app, arguments={"value": 2}, path="/decrement")
    assert response["result"] == 0


def test_app_cancellation(test_app: str, test_cancellable_app: str):
    request_handle = apps.submit(
        test_cancellable_app, arguments={"lhs": 1, "rhs": 2, "wait_time": 6}
    )

    while True:
        status = request_handle.status()
        time.sleep(0.05)
        if isinstance(status, apps.InProgress):
            # The app is running
            break

    # cancel the request
    request_handle.cancel()

    # should still finish successfully and return 499
    with pytest.raises(HTTPStatusError) as e:
        request_handle.get()
    assert e.value.response.status_code == 499

    # normal app should just ignore the cancellation
    request_handle = apps.submit(
        test_app, arguments={"lhs": 1, "rhs": 2, "wait_time": 6}
    )

    while True:
        status = request_handle.status()
        time.sleep(0.05)
        if isinstance(status, apps.InProgress):
            # The app is running
            break

    # cancel the request
    request_handle.cancel()

    response = request_handle.get()
    assert response == {"result": 3}


def test_app_disconnect_behavior(test_app: str, test_cancellable_app: str):
    with pytest.raises(HTTPStatusError) as e:
        apps.run(
            test_cancellable_app,
            arguments={"lhs": 1, "rhs": 2, "wait_time": 6},
            path="/well-handled",
        )
    assert (
        e.value.response.status_code == 504
    ), "Expected Gateway Timeout even though the app handled it"

    # and running it again shows the app "handled" it
    response = apps.run(
        test_cancellable_app,
        arguments={"lhs": 1, "rhs": 2, "wait_time": 1},
        path="/well-handled",
    )
    assert response == {"result": 3}

    # vs on an unhandled one

    with pytest.raises(HTTPStatusError) as e:
        apps.run(
            test_cancellable_app,
            arguments={"lhs": 1, "rhs": 2, "wait_time": 6},
        )
    assert (
        e.value.response.status_code == 504
    ), "Expected Gateway Timeout even though the app handled it"


@pytest.mark.xfail(
    reason="Temporary disabled while investigating backend issue. Ping @efiop"
)
@pytest.mark.flaky(max_runs=3)
def test_app_client_async(test_sleep_app: str):
    handle = apps.submit(test_sleep_app, arguments={"wait_time": 1})
    with pytest.raises(HTTPStatusError) as e:
        # Not yet completed
        handle.fetch_result()

    assert e.value.response.status_code == 400

    # Wait until the app is completed
    assert handle.get() == {"slept": True}

    # New request
    handle = apps.submit(test_sleep_app, arguments={"wait_time": 5})

    for event in handle.iter_events(logs=True):
        assert isinstance(event, (apps.Queued, apps.InProgress))
        if isinstance(event, apps.InProgress) and event.logs:
            logs = [log["message"] for log in event.logs]
            assert "sleeping..." in logs
        elif isinstance(event, apps.Queued):
            assert event.position == 0

    for _ in range(10):
        status = handle.status(logs=True)
        assert isinstance(status, apps.Completed)
        if status.logs:
            break

    assert status.logs, "Logs missing from Completed status"
    assert any("sleeping..." in log["message"] for log in status.logs)

    # It is safe to use fetch_result when we know for a fact the request itself
    # is completed.
    result = handle.fetch_result()

    # .get() can still be used and will return the same value
    get_result = handle.get()
    assert result == get_result
    assert result == {"slept": True}


# If the logging subsystem is not working for some nodes, this test will flake
@pytest.mark.flaky(max_runs=10)
def test_traceback_logs(test_exception_app: AppClient):
    date = (
        datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1)
    ).isoformat()

    with pytest.raises(AppClientError):
        test_exception_app.fail({})

    with httpx.Client(
        base_url=REST_CLIENT.base_url,
        headers=REST_CLIENT.get_headers(),
        timeout=300,
    ) as client:
        # Give some time for logs to propagate through the logging subsystem.
        for _ in range(10):
            time.sleep(2)
            response = client.get(
                REST_CLIENT.base_url + f"/logs/?traceback=true&since={date}"
            )

            logs = response.json()
            if len(logs) > 0:
                break

        assert len(logs) > 0
        for log in logs:
            assert log["message"].count("\n") > 1, "Logs should be multi-line"
            assert (
                '{"traceback":' not in log["message"]
            ), "Logs should not be JSON-wrapped"
            assert (
                "this app is designed to fail" in log["message"]
            ), "Logs should contain the traceback message"


def test_app_openapi_spec_metadata(base_app: Tuple[str, str], user: User):
    app_alias, _ = base_app
    res = app_metadata.sync_detailed(
        app_alias_or_id=app_alias, app_user_id=user.user_id, client=REST_CLIENT
    )

    assert res.status_code == 200, f"Failed to fetch metadata for app {app_alias}"
    assert res.parsed, f"Failed to parse metadata for app {app_alias}"

    metadata = res.parsed.to_dict()
    assert "openapi" in metadata, f"openapi key missing from metadata {metadata}"
    openapi_spec: dict = metadata["openapi"]
    for key in ["openapi", "info", "paths", "components"]:
        assert key in openapi_spec, f"{key} key missing from openapi {openapi_spec}"


def test_app_no_serve_spec_metadata(test_fastapi_app: str):
    # We do not store the openapi spec for apps that do not use serve=True
    user_id, _, app_id = test_fastapi_app.partition("/")
    res = app_metadata.sync_detailed(
        app_alias_or_id=app_id, app_user_id=user_id, client=REST_CLIENT
    )

    assert (
        res.status_code == 200
    ), f"Failed to fetch metadata for app {test_fastapi_app}"
    assert res.parsed, f"Failed to parse metadata for app {test_fastapi_app}"

    metadata = res.parsed.to_dict()
    assert (
        "openapi" not in metadata
    ), f"openapi should not be present in metadata {metadata}"


def test_404_response(test_app: str, request: pytest.FixtureRequest):
    with pytest.raises(HTTPStatusError, match="Path /.*other not found"):
        apps.run(test_app, path="/other", arguments={"lhs": 1, "rhs": 2})


def test_app_no_auth():
    # This will just pass for users with shared apps access
    app_alias = str(uuid.uuid4()) + "-alias"
    with pytest.raises(api.FalServerlessError, match="Must specify auth_mode"):
        addition_app.host.register(
            func=addition_app.func,
            options=addition_app.options,
            # random enough
            application_name=app_alias,
        )


def test_app_deploy_scale(host: api.FalServerlessHost):
    from dataclasses import replace

    app_alias = str(uuid.uuid4()) + "-alias"
    app_revision = addition_app.host.register(
        func=addition_app.func,
        options=addition_app.options,
        application_name=app_alias,
        application_auth_mode="private",
    )

    options = replace(
        addition_app.options, host={**addition_app.options.host, "max_multiplexing": 30}
    )
    kwargs = dict(
        func=addition_app.func,
        options=options,
        application_name=app_alias,
        application_auth_mode="private",
    )

    app_revision = addition_app.host.register(**kwargs, scale=False)

    with host._connection as client:
        res = client.list_aliases()
        found = next(filter(lambda alias: alias.alias == app_alias, res), None)
        assert found, f"Could not find app {app_alias} in {res}"
        assert found.revision == app_revision
        assert found.max_multiplexing == 1

    app_revision = addition_app.host.register(**kwargs, scale=True)

    with host._connection as client:
        res = client.list_aliases()
        found = next(filter(lambda alias: alias.alias == app_alias, res), None)
        assert found, f"Could not find app {app_alias} in {res}"
        assert found.revision == app_revision
        assert found.max_multiplexing == 30


# List aliases is taking long
@pytest.mark.timeout(600)
def test_app_update_app(base_app: Tuple[str, str]):
    app_alias, app_revision = base_app

    host: api.FalServerlessHost = addition_app.host  # type: ignore
    with host._connection as client:
        # Get the registered values
        res = client.list_aliases()
        found = next(filter(lambda alias: alias.alias == app_alias, res), None)
        assert found, f"Could not find app {app_alias} in {res}"
        assert found.revision == app_revision

    with host._connection as client:
        new_keep_alive = found.keep_alive + 1
        new_max_concurrency = found.max_concurrency + 1
        new_max_multiplexing = found.max_multiplexing + 1

        res = client.update_application(
            application_name=app_alias,
            keep_alive=new_keep_alive,
            max_concurrency=new_max_concurrency,
            max_multiplexing=new_max_multiplexing,
        )
        assert res.alias == app_alias
        assert res.keep_alive == new_keep_alive
        assert res.max_concurrency == new_max_concurrency
        assert res.max_multiplexing == new_max_multiplexing

    with host._connection as client:
        new_keep_alive = new_keep_alive + 1
        res = client.update_application(
            application_name=app_alias,
            keep_alive=new_keep_alive,
        )
        assert res.alias == app_alias
        assert res.keep_alive == new_keep_alive
        assert res.max_concurrency == new_max_concurrency
        assert res.max_multiplexing == new_max_multiplexing

    with host._connection as client:
        new_max_concurrency = new_max_concurrency - 1
        res = client.update_application(
            application_name=app_alias,
            max_concurrency=new_max_concurrency,
        )
        assert res.alias == app_alias
        assert res.keep_alive == new_keep_alive
        assert res.max_concurrency == new_max_concurrency
        assert res.max_multiplexing == new_max_multiplexing


# List aliases is taking long
@pytest.mark.timeout(600)
def test_app_set_delete_alias(base_app: Tuple[str, str]):
    app_alias, app_revision = base_app

    host: api.FalServerlessHost = addition_app.host  # type: ignore

    with host._connection as client:
        # Get the registered values
        res = client.list_aliases()
        found = next(filter(lambda alias: alias.alias == app_alias, res), None)
        assert found, f"Could not find app {app_alias} in {res}"
        assert found.revision == app_revision
        assert found.auth_mode == "private"

    with host._connection as client:
        # Get the registered values
        res = client.create_alias(app_alias, app_revision, "public")

    with host._connection as client:
        # Get the registered values
        res = client.list_aliases()
        found = next(filter(lambda alias: alias.alias == app_alias, res), None)
        assert found, f"Could not find app {app_alias} in {res}"
        assert found.revision == app_revision
        assert found.auth_mode == "public"

    with host._connection as client:
        res = client.delete_alias(alias=app_alias)
        assert res == app_revision

    with host._connection as client:
        # Get the registered values
        res = client.list_aliases()
        found = next(filter(lambda alias: alias.alias == app_alias, res), None)
        assert not found, f"Found app {app_alias} in {res} after deletion"


@pytest.mark.flaky(max_runs=3)
def test_realtime_connection(test_realtime_app):
    response = apps.run(test_realtime_app, arguments={"prompt": "a cat"})
    assert response["text"] == "a cat"

    with apps._connect(test_realtime_app) as connection:
        for _ in range(3):
            response = connection.run({"prompt": "a cat"})
            assert response["text"] == "a cat"

    with apps._connect(test_realtime_app, path="/realtime/batched") as connection:
        connection.send({"prompt": "keep busy"})
        time.sleep(0.1)

        for prompt in range(10):
            connection.send({"prompt": str(prompt)})

        assert connection.recv()["texts"] == ["keep busy"]

        received_prompts = set()
        batch_sizes = []
        while len(received_prompts) < 10:
            response = connection.recv()
            received_prompts.update(response["texts"])
            batch_sizes.append(len(response["texts"]))

        assert len(received_prompts) == 10
        assert batch_sizes == [4, 4, 2]


@contextmanager
def delete_workflow_on_exit(client: httpx.Client, workflow_url: str):
    try:
        yield
    finally:
        client.delete(workflow_url)


def test_workflows(test_app: str):
    workflow = Workflow(
        name="test_workflow_" + secrets.token_hex(),
        input_schema={},
        output_schema={},
    )
    # (lhs + rhs) + (lhs + rhs)
    lhs = workflow.run(
        test_app,
        input={
            "lhs": workflow.input.lhs,
            "rhs": workflow.input.rhs,
        },
    )
    rhs = workflow.run(
        test_app,
        input={
            "lhs": workflow.input.lhs,
            "rhs": workflow.input.rhs,
        },
    )
    out = workflow.run(
        test_app,
        input={
            "lhs": lhs.result,
            "rhs": rhs.result,
        },
    )
    workflow.set_output({"result": out.result})
    workflow_id = workflow.publish(title="Test Workflow", is_public=False)

    # Test the underlying app
    data = fal.apps.run(test_app, arguments={"lhs": 2, "rhs": 3})
    assert data["result"] == 5

    with httpx.Client(
        base_url=REST_CLIENT.base_url,
        headers=REST_CLIENT.get_headers(),
        timeout=300,
    ) as client:
        with delete_workflow_on_exit(
            client, REST_CLIENT.base_url + "/workflows/" + workflow_id
        ):
            data = fal.apps.run(
                "workflows/" + workflow_id, arguments={"lhs": 2, "rhs": 3}
            )
            assert data["result"] == 10


def test_app_exceptions(test_exception_app: AppClient):
    with pytest.raises(AppClientError) as app_exc:
        test_exception_app.app_exception({})

    assert app_exc.value.status_code == 401

    with pytest.raises(AppClientError) as field_exc:
        test_exception_app.field_exception({"lhs": 1, "rhs": "2"})

    assert field_exc.value.status_code == 422

    with pytest.raises(AppClientError) as cuda_exc:
        test_exception_app.cuda_exception({})

    assert cuda_exc.value.status_code == _CUDA_OOM_STATUS_CODE
    assert _CUDA_OOM_MESSAGE in cuda_exc.value.message

    with pytest.raises(AppClientError) as cuda_exc:
        test_exception_app.cuda_exception_2({})

    assert cuda_exc.value.status_code == _CUDA_OOM_STATUS_CODE
    assert _CUDA_OOM_MESSAGE in cuda_exc.value.message

    with pytest.raises(AppClientError) as cuda_exc:
        test_exception_app.cuda_exception_3({})

    assert cuda_exc.value.status_code == _CUDA_OOM_STATUS_CODE
    assert _CUDA_OOM_MESSAGE in cuda_exc.value.message


def test_pydantic_validation_billing(test_pydantic_validation_error: AppClient):
    with httpx.Client() as httpx_client:
        url = test_pydantic_validation_error.url + "/increment"
        response = httpx_client.post(
            url,
            json={"value": "this-is-not-an-integer"},
            timeout=30,
        )

        assert response.status_code == 422
        assert response.headers.get("x-fal-billable-units") == "0"


def test_field_exception_billing(test_exception_app: AppClient):
    with httpx.Client() as httpx_client:
        url = test_exception_app.url + "/field-exception"
        response = httpx_client.post(
            url,
            json={"lhs": 1, "rhs": 2},
            timeout=30,
        )

        assert response.status_code == 422
        # For errors raised on runtime, developers should be handling the billing.
        # Therefore not adding billing units.
        assert not hasattr(response.headers, "x-fal-billable-units")


def test_kill_runner(host: api.FalServerlessHost, test_sleep_app: str):
    handle = apps.submit(test_sleep_app, arguments={"wait_time": 10})

    while True:
        status = handle.status()
        if isinstance(status, apps.InProgress):
            break
        elif isinstance(status, apps.Queued):
            time.sleep(1)
        else:
            raise Exception(f"Failed to start the app: {status}")

    with host._connection as client:
        with pytest.raises(Exception) as e:
            client.kill_runner("1234567890")

        assert "not found" in str(e).lower()

        _, _, app_alias = test_sleep_app.partition("/")
        runners = client.list_alias_runners(app_alias)
        assert len(runners) == 1

        client.kill_runner(runners[0].runner_id)

        runners = client.list_alias_runners(app_alias)
        assert len(runners) == 0


def test_container_app_client(test_container_app: str):
    response = apps.run(test_container_app, arguments={"lhs": 1, "rhs": 2})
    assert response["result"] == 3


def test_container_build_args_app_client(test_container_build_args_app: str):
    response = apps.run(test_container_build_args_app, {})
    assert response == "built with build args"


class HintsApp(fal.App, keep_alive=300, max_concurrency=1):
    machine_type = "S"

    def provide_hints(self) -> List[str]:
        return ["é", "😀"]

    @fal.endpoint("/add")
    def add(self, input: Input) -> Output:
        return Output(result=input.lhs + input.rhs)


def test_hints_encoding():
    """
    Make sure that hints that can't be encoded in latin-1 don't crash the app
    https://github.com/encode/starlette/blob/a766a58d14007f07c0b5782fa78cdc370b892796/starlette/datastructures.py#L568
    """
    with AppClient.connect(HintsApp) as client:
        with httpx.Client() as httpx_client:
            url = client.url + "/add"
            resp = httpx_client.post(
                url,
                json={"lhs": 1, "rhs": 2},
                timeout=30,
            )
            assert resp.is_success
            assert resp.json()["result"] == 3
