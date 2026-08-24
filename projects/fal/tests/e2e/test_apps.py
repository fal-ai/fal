import asyncio
import json
import os
import secrets
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta, timezone
from typing import (
    AsyncIterator,
    Callable,
    ContextManager,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import httpx
import pytest
from fastapi import Request, WebSocket
from httpx import HTTPStatusError
from isolate.backends.common import active_python
from openapi_fal_rest.api.applications import app_metadata
from openapi_fal_rest.client import Client
from pydantic import BaseModel
from pydantic import __version__ as pydantic_version
from websockets.sync import client as ws_client

import fal
import fal.api as api
from fal import apps
from fal.api.deploy import User, _get_user
from fal.app import AppClient, AppClientError, wrap_app
from fal.container import ContainerImage
from fal.exceptions import (
    AppException,
    FalServerlessException,
    FieldException,
    RequestCancelledException,
)
from fal.exceptions.gpu import _CUDA_OOM_MESSAGE, _GPU_ERROR_STATUS_CODE
from fal.ref import get_current_app
from fal.sdk import ApplicationHealthCheckConfig, RunnerState, get_credentials
from fal.toolkit.utils.endpoint import cancel_on_disconnect
from fal.workflows import Workflow


@pytest.fixture(scope="module")
def rest_client() -> Generator[Client, None, None]:
    client = api.client.SyncServerlessClient()
    yield client._create_rest_client()


class Input(BaseModel):
    lhs: int
    rhs: int
    wait_time: int = 0


class StatefulInput(BaseModel):
    value: int


class FieldInput(BaseModel):
    value: Union[int, str, float]

    class Config:
        smart_union = True


class Output(BaseModel):
    result: int


class FailInput(BaseModel):
    marker: str


actual_python = active_python()
T = TypeVar("T")


def _auth_headers() -> Dict[str, str]:
    return get_credentials().to_headers()


def _wait_until(
    fetch: Callable[[], T],
    predicate: Callable[[T], bool],
    *,
    timeout: float,
    description: str,
    interval: float = 0.1,
) -> T:
    deadline = time.monotonic() + timeout

    while True:
        value = fetch()
        if predicate(value):
            return value

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(f"Timed out waiting for {description}: {value!r}")
        time.sleep(min(interval, remaining))


def _wait_for_request_status(
    handle,
    expected_status,
    *,
    timeout: float = 60,
    logs: bool = False,
):
    def fetch_status():
        status = handle.status(logs=logs)
        if isinstance(status, apps.Completed) and not isinstance(
            status, expected_status
        ):
            raise AssertionError(
                f"Request completed before reaching {expected_status}: {status!r}"
            )
        return status

    return _wait_until(
        fetch_status,
        lambda status: isinstance(status, expected_status),
        timeout=timeout,
        description=f"request status {expected_status}",
    )


def _cancel_and_wait(handle, *, timeout: float = 30):
    status = handle.status()
    if isinstance(status, apps.Completed):
        return status

    try:
        handle.cancel()
    except HTTPStatusError:
        status = handle.status()
        if isinstance(status, apps.Completed):
            return status
        raise

    return _wait_for_request_status(handle, apps.Completed, timeout=timeout)


def _wait_for_alias_runners(
    client,
    app_alias: str,
    predicate,
    *,
    timeout: float = 45,
):
    return _wait_until(
        lambda: client.list_alias_runners(app_alias),
        predicate,
        timeout=timeout,
        interval=0.5,
        description=f"runner state for {app_alias}",
    )


def _wait_for_alias_revision(
    client,
    app_alias: str,
    app_revision: str,
    *,
    timeout: float = 60,
):
    def fetch_alias():
        return next(
            (alias for alias in client.list_aliases() if alias.alias == app_alias),
            None,
        )

    return _wait_until(
        fetch_alias,
        lambda alias: alias is not None and alias.revision == app_revision,
        timeout=timeout,
        interval=0.5,
        description=f"alias {app_alias} to point to revision {app_revision}",
    )


def _is_alias_not_found_response(response: httpx.Response, app_alias: str) -> bool:
    if response.status_code != 404:
        return False

    try:
        detail = response.json().get("detail", "")
    except ValueError:
        return False
    return detail in {
        f"Application {app_alias!r} not found",
        f'Application "{app_alias}" not found',
    }


def _wait_for_stable_alias(
    fetch_response: Callable[[float], Optional[httpx.Response]],
    app_alias: str,
    *,
    timeout: float,
    description: str,
    stable_for: float = 5,
) -> httpx.Response:
    # One updated replica can succeed while another still has stale alias state.
    deadline = time.monotonic() + timeout
    recognized_at = None
    response = None

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(f"Timed out waiting for {description}: {response!r}")

        response = fetch_response(remaining)
        if time.monotonic() >= deadline:
            raise AssertionError(f"Timed out waiting for {description}: {response!r}")
        if response is None or _is_alias_not_found_response(response, app_alias):
            recognized_at = None
        elif recognized_at is None:
            recognized_at = time.monotonic()
        elif time.monotonic() - recognized_at >= stable_for:
            return response

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(f"Timed out waiting for {description}: {response!r}")
        time.sleep(min(0.5, remaining))


def _wait_for_queue_alias(
    app_alias: str,
    queue_url: str,
    *,
    timeout: float = 60,
):
    with httpx.Client(headers=_auth_headers()) as client:

        def fetch_response(remaining: float):
            # Alias resolution precedes validation of this query parameter, whose
            # invalid value makes the gateway return before enqueuing a request.
            response = client.post(
                queue_url,
                params={"fal_max_queue_length": "readiness-probe"},
                json={},
                timeout=min(5, remaining),
            )
            if _is_alias_not_found_response(response, app_alias):
                return response

            if response.status_code != 400:
                response.raise_for_status()
                raise AssertionError(f"Unexpected queue readiness response: {response}")

            data = response.json()
            error = data.get("error", "")
            if data.get("request_id") is not None or not error.startswith(
                "Invalid fal_max_queue_length"
            ):
                raise AssertionError(f"Unexpected queue readiness response: {data}")
            return response

        _wait_for_stable_alias(
            fetch_response,
            app_alias,
            timeout=timeout,
            description=f"queue gateway to recognize alias {app_alias}",
        )


def _wait_for_run_alias(
    app_alias: str,
    run_url: str,
    *,
    timeout: float = 60,
):
    with httpx.Client(headers=_auth_headers()) as client:

        def fetch_response(remaining: float):
            try:
                return client.get(f"{run_url}/health", timeout=min(5, remaining))
            except httpx.TimeoutException:
                return None

        _wait_for_stable_alias(
            fetch_response,
            app_alias,
            timeout=timeout,
            description=f"run gateway to recognize alias {app_alias}",
        )


GIT_REVISION_SHORT_HASH = (
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


@fal.function(
    kind="container",
    image=ContainerImage.from_dockerfile_str(
        f"FROM python:{actual_python}-slim\n# {GIT_REVISION_SHORT_HASH}",
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
        f"FROM python:{actual_python}-slim\n# {GIT_REVISION_SHORT_HASH}",
    ),
    keep_alive=60,
    machine_type="S",
    serve=True,
    max_concurrency=1,
    force_env_build=False,
)
def container_cache_enabled_app(input: Input) -> Output:
    print("starting...")
    for _ in range(input.wait_time):
        print("sleeping...")
        time.sleep(1)

    return Output(result=input.lhs + input.rhs)


@fal.function(
    kind="container",
    image=ContainerImage.from_dockerfile_str(
        f"""FROM python:{actual_python}-slim\n# {GIT_REVISION_SHORT_HASH}
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


GREET_SERVER_OPENAPI = {
    "openapi": "3.0.3",
    "info": {"title": "Greet Server", "version": "1.0.0"},
    "paths": {
        "/health": {
            "get": {
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                }
            }
        },
        "/greet": {
            "post": {
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/GreetInput"}
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/GreetOutput"}
                            }
                        },
                    }
                },
            }
        },
    },
    "components": {
        "schemas": {
            "GreetInput": {
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
            },
            "GreetOutput": {
                "type": "object",
                "properties": {"greeting": {"type": "string"}},
            },
        }
    },
}


@fal.function(
    keep_alive=300,
    requirements=["fastapi", "uvicorn", f"pydantic=={pydantic_version}"],
    machine_type="S",
    max_concurrency=1,
    exposed_port=8080,
    metadata={"openapi": GREET_SERVER_OPENAPI},
)
def greet_server_app():
    """@fal.function with exposed_port + user-supplied openapi declaring /health.

    Regression guard for the bring-your-own-server + custom openapi
    deployment shape.
    """
    from fastapi import FastAPI
    from pydantic import BaseModel
    from uvicorn import run

    fastapi_app = FastAPI()

    class GreetInput(BaseModel):
        name: str

    class GreetOutput(BaseModel):
        greeting: str

    @fastapi_app.get("/health")
    def health():
        return {"status": "ok"}

    @fastapi_app.post("/greet")
    def greet(req: GreetInput) -> GreetOutput:
        return GreetOutput(greeting=f"Hello, {req.name}!")

    run(fastapi_app, host="0.0.0.0", port=8080)


@fal.function(
    keep_alive=300,
    requirements=["fastapi", "uvicorn", f"pydantic=={pydantic_version}"],
    machine_type="S",
    max_concurrency=1,
    exposed_port=8080,
    health_check_config=ApplicationHealthCheckConfig(
        path="/ready",
        start_period_seconds=1,
        timeout_seconds=5,
        failure_threshold=1,
        call_regularly=True,
    ),
)
def custom_health_path_app():
    """@fal.function with user-declared health_check_config at a non-default path.

    Regression guard for the case where users bring their own server
    and want the platform to use a specific health endpoint instead
    of the default /health.
    """
    from fastapi import FastAPI
    from pydantic import BaseModel
    from uvicorn import run

    fastapi_app = FastAPI()

    class GreetInput(BaseModel):
        name: str

    class GreetOutput(BaseModel):
        greeting: str

    @fastapi_app.get("/ready")
    def ready():
        return {"status": "ok"}

    @fastapi_app.post("/greet")
    def greet(req: GreetInput) -> GreetOutput:
        return GreetOutput(greeting=f"Hello, {req.name}!")

    run(fastapi_app, host="0.0.0.0", port=8080)


@fal.function(
    keep_alive=300,
    requirements=["fastapi", "uvicorn", f"pydantic=={pydantic_version}"],
    machine_type="S",
    max_concurrency=1,
    exposed_port=8080,
    health_check_config=ApplicationHealthCheckConfig(
        path="/ready",
        start_period_seconds=1,
        timeout_seconds=5,
        failure_threshold=1,
        call_regularly=True,
    ),
)
def health_override_fn():
    """@fal.function declaring health at /ready, with a deliberately broken
    /health endpoint that returns 502.

    Regression guard for "user-declared health endpoint wins over the
    default /health path." If the deployment reaches ready and /greet
    responds, the platform correctly used /ready (the broken /health
    must have been ignored).
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from uvicorn import run

    fastapi_app = FastAPI()

    class GreetInput(BaseModel):
        name: str

    class GreetOutput(BaseModel):
        greeting: str

    @fastapi_app.get("/ready")
    def ready():
        return {"status": "ok"}

    @fastapi_app.get("/health")
    def broken_health():
        raise HTTPException(status_code=502, detail="intentionally broken")

    @fastapi_app.post("/greet")
    def greet(req: GreetInput) -> GreetOutput:
        return GreetOutput(greeting=f"Hello, {req.name}!")

    run(fastapi_app, host="0.0.0.0", port=8080)


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


class QueueBlockingApp(fal.App, keep_alive=300, max_concurrency=1, max_multiplexing=1):
    """
    App for testing start_timeout with queue blocking.

    With max_concurrency=1 and max_multiplexing=1, only ONE request can be
    processed at a time. Additional requests must wait in the queue.
    """

    machine_type = "XS"

    @fal.endpoint("/")
    async def sleep(self, input: SleepInput) -> SleepOutput:
        for i in range(input.wait_time):
            print(f"sleeping {i + 1}/{input.wait_time}...", flush=True)
            await asyncio.sleep(1)
        return SleepOutput(slept=True)


class ExceptionApp(fal.App, keep_alive=300, max_concurrency=1):
    machine_type = "XS"

    @fal.endpoint("/fail")
    def fail(self, input: FailInput) -> Output:
        raise Exception(f"this app is designed to fail! {input.marker}")

    @fal.endpoint("/app-exception")
    def app_exception(self) -> Output:
        raise AppException(message="this app is designed to fail", status_code=401)

    @fal.endpoint("/field-exception")
    def field_exception(self, input: Input) -> Output:
        raise FieldException(
            field="rhs",
            message="rhs must be an integer",
        )

    @fal.endpoint("/field-exception-units")
    def field_exception_units(self, input: FieldInput) -> Output:
        raise FieldException(
            field="value",
            message="value must be a valid value",
            billable_units=input.value,
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
    skip_retry_conditions = ["timeout"]
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


class HealthOverrideApp(fal.App, keep_alive=300, max_concurrency=1):
    """fal.App declaring health at /ready with a broken default /health.

    Regression guard for "user-declared health endpoint wins over the
    default /health path." The default /health endpoint is intentionally
    broken (returns 502 via the overridden ``health()`` method); a custom
    GET /ready route is added so probes succeed. If the deployment reaches
    ready and / responds, the platform correctly used /ready and ignored
    the broken /health.
    """

    machine_type = "S"

    @fal.endpoint("/")
    def run(self, input: Input) -> Output:
        return Output(result=input.lhs + input.rhs)

    @fal.endpoint(
        "/ready",
        health_check=fal.HealthCheck(
            start_period_seconds=1,
            timeout_seconds=5,
            failure_threshold=1,
            call_regularly=True,
        ),
    )
    def ready_post(self) -> Output:
        return Output(result=0)

    def _add_extra_routes(self, app):
        # @fal.endpoint registers POST-only; add a GET /ready route so
        # GET-based probes hit a happy 200.
        super()._add_extra_routes(app)

        @app.get("/ready")
        def ready_get():
            return {"status": "ok"}

    def health(self):
        # Intentionally broken so /health returns 502.
        from fastapi import HTTPException

        raise HTTPException(status_code=502, detail="intentionally broken")


class RTInput(BaseModel):
    prompt: str

    def can_batch(
        self,
        other: "RTInput",
        current_batch_size: int = 1,
    ) -> bool:
        return "don't batch" not in self.prompt and "don't batch" not in other.prompt


class RTOutput(BaseModel):
    text: str


class RTOutputs(BaseModel):
    texts: List[str]


def json_encode_message(message):
    return json.dumps(message, separators=(",", ":")).encode("utf-8")


def json_decode_message(message: bytes):
    return json.loads(message.decode("utf-8"))


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

    @fal.realtime("/realtime/server-streaming", buffering=10)
    async def generate_rt_server_streaming(
        self, input: RTInput
    ) -> AsyncIterator[RTOutput]:
        for idx in range(3):
            yield RTOutput(text=f"{input.prompt}:{idx}")

    @fal.realtime("/realtime/server-streaming-sync", buffering=10)
    def generate_rt_server_streaming_sync(self, input: RTInput) -> Iterator[RTOutput]:
        for idx in range(3):
            yield RTOutput(text=f"{input.prompt}:{idx}")

    @fal.realtime("/realtime/client-streaming", session_timeout=1)
    async def generate_rt_client_streaming(
        self, inputs: AsyncIterator[RTInput]
    ) -> RTOutputs:
        prompts: List[str] = []
        async for item in inputs:
            prompts.append(item.prompt)
        return RTOutputs(texts=prompts)

    @fal.realtime("/realtime/bidi")
    async def generate_rt_bidi(
        self, inputs: AsyncIterator[RTInput]
    ) -> AsyncIterator[RTOutput]:
        async for item in inputs:
            yield RTOutput(text=f"echo:{item.prompt}")

    @fal.realtime(
        "/realtime/json",
        encode_message=json_encode_message,
        decode_message=json_decode_message,
    )
    def generate_rt_json(self, input: RTInput) -> RTOutput:
        return RTOutput(text=input.prompt)

    @fal.realtime("/realtime/batched", buffering=10, max_batch_size=4)
    def generate_rt_batched(self, input: RTInput, *inputs: RTInput) -> RTOutputs:
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
def user(rest_client: Client) -> Generator[User, None, None]:
    user = _get_user(rest_client)
    yield user


@pytest.fixture(scope="module")
def register_app(
    host: api.FalServerlessHost,
    make_tmp_app_name: Callable[[str], str],
) -> Callable[
    [
        Union[api.ServedIsolatedFunction, api.IsolatedFunction],
        str,
    ],
    ContextManager[Tuple[str, str]],
]:
    @contextmanager
    def _register_app(
        app: Union[api.ServedIsolatedFunction, api.IsolatedFunction],
        suffix: str = "",
    ):
        app_alias = make_tmp_app_name(suffix)
        result = host.register(
            func=app.func,
            options=app.options,
            application_name=app_alias,
            application_auth_mode="private",
            deployment_strategy="recreate",
        )

        assert result
        assert result.result
        assert result.service_urls
        app_revision = result.result.application_id

        try:
            with host._connection as client:
                _wait_for_alias_revision(client, app_alias, app_revision)
            _wait_for_queue_alias(app_alias, result.service_urls.queue)
            _wait_for_run_alias(app_alias, result.service_urls.run)
            yield app_alias, app_revision
        finally:
            with host._connection as client:
                client.delete_alias(app_alias)

    return _register_app


@pytest.fixture()
def base_app(register_app):
    # running apps without aliases is no longer supported
    # so we need to create an alias for the app
    with register_app(addition_app, "base") as (
        app_alias,
        app_revision,
    ):
        yield app_alias, app_revision


@pytest.fixture(scope="module")
def test_app(
    user: User,
    register_app,
):
    with register_app(addition_app, "addition") as (
        app_alias,
        _,
    ):
        yield f"{user.username}/{app_alias}"


@pytest.fixture()
def test_container_app(
    user: User,
    register_app,
):
    with register_app(container_addition_app, "container") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture()
def test_container_build_args_app(
    user: User,
    register_app,
):
    with register_app(container_build_args_app, "build-args") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture()
def test_greet_server_app(
    user: User,
    register_app,
):
    with register_app(greet_server_app, "greet-server") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture()
def test_custom_health_path_app(
    user: User,
    register_app,
):
    with register_app(custom_health_path_app, "custom-health") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture()
def test_health_override_fn(
    user: User,
    register_app,
):
    with register_app(health_override_fn, "health-override-fn") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture()
def test_health_override_app(
    user: User,
    register_app,
):
    health_override_app = wrap_app(HealthOverrideApp)
    with register_app(health_override_app, "health-override-app") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture(scope="module")
def test_stateful_app(
    user: User,
    register_app,
):
    stateful_app = wrap_app(StatefulAdditionApp)
    with register_app(stateful_app, "stateful") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture()
def test_cancellable_app(
    user: User,
    register_app,
):
    cancellable_app = wrap_app(CancellableApp)
    with register_app(cancellable_app, "cancellable") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture(scope="module")
def test_exception_app():
    with AppClient.connect(ExceptionApp) as client:
        yield client


@pytest.fixture()
def test_sleep_app(
    user: User,
    register_app,
):
    sleep_app = wrap_app(SleepApp)
    with register_app(sleep_app, "sleep") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture()
def test_queue_blocking_app(
    user: User,
    register_app,
):
    queue_blocking_app = wrap_app(QueueBlockingApp)
    with register_app(queue_blocking_app, "queue-blocking") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.fixture(scope="module")
def test_realtime_app(
    user: User,
    register_app,
):
    realtime_app = wrap_app(RealtimeApp)
    with register_app(realtime_app, "realtime") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


def test_broken_app_failure():
    with pytest.raises(FalServerlessException) as e:
        wrap_app(BrokenApp)

    assert "Failed to generate OpenAPI" in str(e)


@pytest.mark.xdist_group(name="addition-app")
def test_app_client(test_app: str):
    response = apps.run(test_app, arguments={"lhs": 1, "rhs": 2})
    assert response["result"] == 3

    response = apps.run(test_app, arguments={"lhs": 2, "rhs": 3, "wait_time": 1})
    assert response["result"] == 5


def test_function_with_custom_openapi_health(
    test_greet_server_app: str, rest_client: Client
):
    """@fal.function deployments using exposed_port + custom openapi declaring
    /health: both /greet and /health must be reachable. The platform does
    not persist user-supplied openapi for non-serve apps (parallel to
    test_app_no_serve_spec_metadata), which is documented here."""
    from fal.flags import FAL_RUN_HOST

    # The platform does not store user-supplied openapi for non-serve apps.
    # Same observation as test_app_no_serve_spec_metadata for calculator_app.
    user_id, _, app_id = test_greet_server_app.partition("/")
    res = app_metadata.sync_detailed(
        app_alias_or_id=app_id, app_user_id=user_id, client=rest_client
    )
    assert res.status_code == 200, f"Failed to fetch metadata: {res}"
    assert res.parsed, "Failed to parse metadata"
    metadata = res.parsed.to_dict()
    assert "openapi" not in metadata, f"openapi unexpectedly persisted: {metadata}"

    r = httpx.post(
        f"https://{FAL_RUN_HOST}/{test_greet_server_app}/greet",
        json={"name": "world"},
        headers=_auth_headers(),
        timeout=60,
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"greeting": "Hello, world!"}

    r = httpx.get(
        f"https://{FAL_RUN_HOST}/{test_greet_server_app}/health",
        headers=_auth_headers(),
        timeout=30,
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ok"}


def test_function_with_custom_health_path(test_custom_health_path_app: str):
    """@fal.function with explicit health_check_config at a non-default path:
    both the user-declared health endpoint (/ready) and the app's main
    endpoint (/greet) must be reachable through the platform."""
    from fal.flags import FAL_RUN_HOST

    r = httpx.post(
        f"https://{FAL_RUN_HOST}/{test_custom_health_path_app}/greet",
        json={"name": "world"},
        headers=_auth_headers(),
        timeout=60,
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"greeting": "Hello, world!"}

    r = httpx.get(
        f"https://{FAL_RUN_HOST}/{test_custom_health_path_app}/ready",
        headers=_auth_headers(),
        timeout=30,
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ok"}


def test_function_health_override(test_health_override_fn: str):
    """@fal.function declaring health at /ready, with /health intentionally
    returning 502: the platform must respect the user-declared health
    endpoint and ignore the broken default /health.

    If the deployment reaches ready and /greet responds, the platform used
    /ready (otherwise /health's 502 would have blocked readiness).
    """
    from fal.flags import FAL_RUN_HOST

    # main endpoint works → runner reached ready → platform probed /ready
    r = httpx.post(
        f"https://{FAL_RUN_HOST}/{test_health_override_fn}/greet",
        json={"name": "world"},
        headers=_auth_headers(),
        timeout=60,
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"greeting": "Hello, world!"}

    # /ready is the user-declared health endpoint
    r = httpx.get(
        f"https://{FAL_RUN_HOST}/{test_health_override_fn}/ready",
        headers=_auth_headers(),
        timeout=30,
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ok"}

    # /health is intentionally broken — its 502 is observable but didn't
    # block readiness because the platform used /ready instead
    r = httpx.get(
        f"https://{FAL_RUN_HOST}/{test_health_override_fn}/health",
        headers=_auth_headers(),
        timeout=30,
    )
    assert r.status_code == 502, r.text


def test_app_health_override(test_health_override_app: str):
    """fal.App declaring health at /ready, with the default /health endpoint
    intentionally broken (returns 502): the platform must respect the
    user-declared health endpoint and ignore /health.

    If the deployment reaches ready and / responds, the platform used /ready
    (otherwise /health's 502 would have blocked readiness).
    """
    from fal.flags import FAL_RUN_HOST

    # main endpoint works → runner reached ready → platform probed /ready
    r = httpx.post(
        f"https://{FAL_RUN_HOST}/{test_health_override_app}/",
        json={"lhs": 1, "rhs": 2},
        headers=_auth_headers(),
        timeout=60,
    )
    assert r.status_code == 200, r.text
    assert r.json()["result"] == 3

    # /ready is the user-declared health endpoint
    r = httpx.get(
        f"https://{FAL_RUN_HOST}/{test_health_override_app}/ready",
        headers=_auth_headers(),
        timeout=30,
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ok"}

    # default /health is intentionally broken — observable 502, didn't
    # block readiness
    r = httpx.get(
        f"https://{FAL_RUN_HOST}/{test_health_override_app}/health",
        headers=_auth_headers(),
        timeout=30,
    )
    assert r.status_code == 502, r.text


@pytest.mark.xdist_group(name="addition-app")
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


@pytest.mark.xdist_group(name="stateful-app")
def test_app_client_path_included_in_app_id(test_stateful_app: str):
    response = apps.run(test_stateful_app + "/reset", arguments={})
    assert response["result"] == 0

    response = apps.run(test_stateful_app + "/increment", arguments={"value": 3})
    assert response["result"] == 3

    # if put in path we do not need to prefix with /
    response = apps.run(test_stateful_app, arguments={"value": 3}, path="increment")
    assert response["result"] == 6


@pytest.mark.xdist_group(name="stateful-app")
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


@pytest.mark.xdist_group(name="addition-app")
def test_app_cancellation(test_app: str, test_cancellable_app: str):
    request_handle = apps.submit(
        test_cancellable_app, arguments={"lhs": 1, "rhs": 2, "wait_time": 6}
    )

    _wait_for_request_status(request_handle, apps.InProgress)

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

    _wait_for_request_status(request_handle, apps.InProgress)

    # cancel the request
    request_handle.cancel()

    response = request_handle.get()
    assert response == {"result": 3}


def test_app_disconnect_behavior(test_cancellable_app: str):
    with pytest.raises(HTTPStatusError) as e:
        apps.run(
            test_cancellable_app,
            arguments={"lhs": 1, "rhs": 2, "wait_time": 30},
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
            arguments={"lhs": 1, "rhs": 2, "wait_time": 30},
        )
    assert (
        e.value.response.status_code == 504
    ), "Expected Gateway Timeout even though the app handled it"


@pytest.mark.timeout(120)
def test_start_timeout_queue_blocking(test_queue_blocking_app: str):
    """
    Test that start_timeout correctly times out a request waiting in queue.

    Scenario:
    1. Send a long-running request (occupies the only slot)
    2. While it's processing, send a second request with start_timeout=2
    3. The second request should return 504 because it times out waiting in queue
       (before processing starts)
    4. Cancel the blocking request during cleanup
    """
    import fal_client
    from fal_client.client import FalClientHTTPError

    first_handle = apps.submit(test_queue_blocking_app, arguments={"wait_time": 15})

    try:
        _wait_for_request_status(first_handle, apps.InProgress, timeout=60)

        with pytest.raises(FalClientHTTPError) as exc_info:
            fal_client.subscribe(
                test_queue_blocking_app,
                arguments={"wait_time": 1},
                start_timeout=2,
            )
    finally:
        _cancel_and_wait(first_handle)

    # Should get a 504 timeout error
    assert (
        exc_info.value.status_code == 504
    ), f"Expected 504 timeout, got {exc_info.value.status_code}"

    # Verify the timeout type header indicates it was a user timeout
    timeout_type = exc_info.value.response_headers.get("x-fal-request-timeout-type")
    assert timeout_type == "user", f"Expected 'user' timeout type, got {timeout_type}"


@pytest.mark.timeout(120)
def test_app_client_async(test_sleep_app: str):
    handle = apps.submit(test_sleep_app, arguments={"wait_time": 10})
    _wait_for_request_status(handle, apps.InProgress)
    with pytest.raises(HTTPStatusError) as e:
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

    status = _wait_until(
        lambda: handle.status(logs=True),
        lambda current: isinstance(current, apps.Completed) and bool(current.logs),
        timeout=30,
        description="completed request logs",
    )

    assert status.logs, "Logs missing from Completed status"
    assert any("sleeping..." in log["message"] for log in status.logs)

    # It is safe to use fetch_result when we know for a fact the request itself
    # is completed.
    result = handle.fetch_result()

    # .get() can still be used and will return the same value
    get_result = handle.get()
    assert result == get_result
    assert result == {"slept": True}


@pytest.mark.xfail(
    reason="Temporary disabled while investigating backend issue. Ping @efiop"
)
@pytest.mark.xdist_group(name="exception-app")
def test_traceback_logs(test_exception_app: AppClient, rest_client: Client):
    marker = f"traceback-{secrets.token_hex(8)}"
    date = (
        datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)
    ).isoformat()

    with pytest.raises(AppClientError):
        test_exception_app.fail({"marker": marker})

    with httpx.Client(
        base_url=rest_client.base_url,
        headers=rest_client.get_headers(),
        timeout=300,
    ) as client:

        def fetch_matching_logs():
            response = client.get(
                rest_client.base_url + f"/logs/?traceback=true&since={date}"
            )
            response.raise_for_status()
            return [log for log in response.json() if marker in log["message"]]

        logs = _wait_until(
            fetch_matching_logs,
            bool,
            timeout=45,
            interval=1,
            description="traceback log propagation",
        )

        for log in logs:
            assert log["message"].count("\n") > 1, "Logs should be multi-line"
            assert (
                '{"traceback":' not in log["message"]
            ), "Logs should not be JSON-wrapped"


@pytest.mark.xdist_group(name="addition-app")
def test_app_openapi_spec_metadata(test_app: str, rest_client: Client):
    app_user_id, _, app_alias = test_app.partition("/")
    res = app_metadata.sync_detailed(
        app_alias_or_id=app_alias, app_user_id=app_user_id, client=rest_client
    )

    assert res.status_code == 200, f"Failed to fetch metadata for app {app_alias}"
    assert res.parsed, f"Failed to parse metadata for app {app_alias}"

    metadata = res.parsed.to_dict()
    assert "openapi" in metadata, f"openapi key missing from metadata {metadata}"
    openapi_spec: dict = metadata["openapi"]
    for key in ["openapi", "info", "paths", "components"]:
        assert key in openapi_spec, f"{key} key missing from openapi {openapi_spec}"


def test_app_no_serve_spec_metadata(
    host: api.FalServerlessHost,
    user: User,
    rest_client: Client,
    make_tmp_app_name: Callable[[str], str],
):
    # We do not store the openapi spec for apps that do not use serve=True
    app_alias = make_tmp_app_name("fastapi")
    result = host.register(
        func=calculator_app.func,
        options=calculator_app.options,
        application_name=app_alias,
        application_auth_mode="private",
        deployment_strategy="recreate",
    )

    assert result
    assert result.result

    try:
        with host._connection as client:
            _wait_for_alias_revision(client, app_alias, result.result.application_id)

        res = app_metadata.sync_detailed(
            app_alias_or_id=app_alias,
            app_user_id=user.username,
            client=rest_client,
        )

        assert (
            res.status_code == 200
        ), f"Failed to fetch metadata for app {user.username}/{app_alias}"
        assert (
            res.parsed
        ), f"Failed to parse metadata for app {user.username}/{app_alias}"

        metadata = res.parsed.to_dict()
        assert (
            "openapi" not in metadata
        ), f"openapi should not be present in metadata {metadata}"
    finally:
        with host._connection as client:
            client.delete_alias(app_alias)


@pytest.mark.xdist_group(name="addition-app")
def test_404_response(test_app: str):
    with pytest.raises(HTTPStatusError, match="Path /.*other not found"):
        apps.run(test_app, path="/other", arguments={"lhs": 1, "rhs": 2})


@pytest.mark.xdist_group(name="exception-app")
def test_404_billable_units(test_exception_app: AppClient):
    """Test that 404 responses include x-fal-billable-units: 0 header."""
    with httpx.Client(headers=_auth_headers()) as httpx_client:
        url = test_exception_app.url + "/non-existent-endpoint"
        response = httpx_client.post(
            url,
            json={},
            timeout=30,
        )

        assert response.status_code == 404
        assert response.headers.get("x-fal-billable-units") == "0"


def test_app_deploy_scale(host: api.FalServerlessHost, register_app):
    from dataclasses import replace

    with register_app(addition_app, "deploy-scale") as (app_alias, _):
        options = replace(
            addition_app.options,
            host={
                **addition_app.options.host,
                "max_multiplexing": 3,
                "max_concurrency": 2,
            },
        )
        kwargs = dict(
            func=addition_app.func,
            options=options,
            application_name=app_alias,
            application_auth_mode="private",
            deployment_strategy="recreate",
        )

        result = addition_app.host.register(**kwargs, scale=False)
        assert result
        assert result.result
        assert result.service_urls
        app_revision = result.result.application_id

        with host._connection as client:
            found = _wait_for_alias_revision(client, app_alias, app_revision)
            # multiplexing is revision-specific
            assert (
                found.max_multiplexing == 3
            ), "Expected max_multiplexing to have changed"
            # max_concurrency is alias-specific
            assert (
                found.max_concurrency == 1
            ), "Expected max_concurrency to stay the same"

        result = addition_app.host.register(**kwargs, scale=True)
        assert result
        assert result.result
        assert result.service_urls
        app_revision = result.result.application_id

        with host._connection as client:
            found = _wait_for_alias_revision(client, app_alias, app_revision)
            # when scaling, all values are updated
            assert found.max_multiplexing == 3
            assert found.max_concurrency == 2


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


@pytest.mark.xdist_group(name="realtime-app")
def test_realtime_connection(test_realtime_app):
    isolated_input = RTInput(prompt="don't batch")
    batchable_input = RTInput(prompt="batchable")
    assert not isolated_input.can_batch(batchable_input)
    assert not batchable_input.can_batch(isolated_input)

    response = apps.run(test_realtime_app, arguments={"prompt": "a cat"})
    assert response["text"] == "a cat"

    with apps._connect(test_realtime_app) as connection:
        for _ in range(3):
            response = connection.run({"prompt": "a cat"})
            assert response["text"] == "a cat"

    with apps._connect(test_realtime_app, path="/realtime/batched") as connection:
        connection.send({"prompt": "don't batch"})
        assert connection.recv()["texts"] == ["don't batch"]

        for prompt in range(10):
            connection.send({"prompt": str(prompt)})

        received_prompts = set()
        batch_sizes = []
        while len(received_prompts) < 10:
            response = connection.recv()
            received_prompts.update(response["texts"])
            batch_sizes.append(len(response["texts"]))

        assert received_prompts == {str(prompt) for prompt in range(10)}
        assert sum(batch_sizes) == 10
        assert all(1 <= batch_size <= 4 for batch_size in batch_sizes)


@pytest.mark.xdist_group(name="realtime-app")
def test_realtime_ws_endpoint(test_realtime_app):
    app_id = apps._backwards_compatible_app_id(test_realtime_app)
    url = apps._REALTIME_URL_FORMAT.format(app_id=app_id) + "/ws"
    creds = get_credentials()

    with ws_client.connect(
        url, additional_headers=creds.to_headers(), open_timeout=90
    ) as ws:
        messages = []
        for _ in range(3):
            payload = ws.recv()
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            messages.append(json.loads(payload))

    assert messages == [{"message": "Hello world!"}] * 3


@pytest.mark.xdist_group(name="realtime-app")
def test_realtime_connection_custom_codec(test_realtime_app):
    with apps._connect(
        test_realtime_app,
        path="/realtime/json",
        encode_message=json_encode_message,
        decode_message=json_decode_message,
    ) as connection:
        response = connection.run({"prompt": "json cat"})
        assert response["text"] == "json cat"


@pytest.mark.xdist_group(name="realtime-app")
def test_realtime_server_streaming_mode(test_realtime_app):
    with apps._connect(
        test_realtime_app, path="/realtime/server-streaming"
    ) as connection:
        connection.send({"prompt": "stream"})
        responses = [connection.recv() for _ in range(3)]
        assert [response["text"] for response in responses] == [
            "stream:0",
            "stream:1",
            "stream:2",
        ]


@pytest.mark.xdist_group(name="realtime-app")
def test_realtime_server_streaming_sync_mode(test_realtime_app):
    with apps._connect(
        test_realtime_app, path="/realtime/server-streaming-sync"
    ) as connection:
        connection.send({"prompt": "stream"})
        responses = [connection.recv() for _ in range(3)]
        assert [response["text"] for response in responses] == [
            "stream:0",
            "stream:1",
            "stream:2",
        ]


@pytest.mark.xdist_group(name="realtime-app")
def test_realtime_client_streaming_mode(test_realtime_app):
    with apps._connect(
        test_realtime_app, path="/realtime/client-streaming"
    ) as connection:
        connection.send({"prompt": "first"})
        connection.send({"prompt": "second"})
        connection.send({"prompt": "third"})
        response = connection.recv()
        assert response["texts"] == ["first", "second", "third"]


@pytest.mark.xdist_group(name="realtime-app")
def test_realtime_bidi_mode(test_realtime_app):
    with apps._connect(test_realtime_app, path="/realtime/bidi") as connection:
        connection.send({"prompt": "one"})
        connection.send({"prompt": "two"})
        assert connection.recv()["text"] == "echo:one"
        assert connection.recv()["text"] == "echo:two"


@contextmanager
def delete_workflow_on_exit(client: httpx.Client, workflow_url: str):
    try:
        yield
    finally:
        client.delete(workflow_url)


@pytest.mark.xdist_group(name="addition-app")
def test_workflows(test_app: str, rest_client: Client):
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
        base_url=rest_client.base_url,
        headers=rest_client.get_headers(),
        timeout=300,
    ) as client:
        with delete_workflow_on_exit(
            client, rest_client.base_url + "/workflows/" + workflow_id
        ):
            data = fal.apps.run(
                "workflows/" + workflow_id, arguments={"lhs": 2, "rhs": 3}
            )
            assert data["result"] == 10


@pytest.mark.xdist_group(name="exception-app")
def test_app_exceptions(test_exception_app: AppClient):
    with pytest.raises(AppClientError) as app_exc:
        test_exception_app.app_exception({})

    assert app_exc.value.status_code == 401

    with pytest.raises(AppClientError) as field_exc:
        test_exception_app.field_exception({"lhs": 1, "rhs": "2"})

    assert field_exc.value.status_code == 422

    assert field_exc.value.headers.get("x-fal-billable-units") is None

    with pytest.raises(AppClientError) as cuda_exc:
        test_exception_app.cuda_exception({})

    assert cuda_exc.value.status_code == _GPU_ERROR_STATUS_CODE
    assert _CUDA_OOM_MESSAGE in cuda_exc.value.message

    with pytest.raises(AppClientError) as cuda_exc:
        test_exception_app.cuda_exception_2({})

    assert cuda_exc.value.status_code == _GPU_ERROR_STATUS_CODE
    assert _CUDA_OOM_MESSAGE in cuda_exc.value.message

    with pytest.raises(AppClientError) as cuda_exc:
        test_exception_app.cuda_exception_3({})

    assert cuda_exc.value.status_code == _GPU_ERROR_STATUS_CODE
    assert _CUDA_OOM_MESSAGE in cuda_exc.value.message


@pytest.mark.xdist_group(name="stateful-app")
def test_pydantic_validation_billing(test_stateful_app: str):
    from fal.flags import FAL_RUN_HOST

    with httpx.Client(headers=_auth_headers()) as httpx_client:
        url = f"https://{FAL_RUN_HOST}/{test_stateful_app}/increment"
        response = httpx_client.post(
            url,
            json={"value": "this-is-not-an-integer"},
            timeout=30,
        )

        assert response.status_code == 422
        assert response.headers.get("x-fal-billable-units") == "0"


@pytest.mark.xdist_group(name="exception-app")
def test_field_exception_billing(test_exception_app: AppClient):
    with httpx.Client(headers=_auth_headers()) as httpx_client:
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


@pytest.mark.xdist_group(name="exception-app")
def test_field_exception_int_billable_units_formatting(test_exception_app: AppClient):
    """Test that int billable_units are formatted without decimal places."""
    with httpx.Client(headers=_auth_headers()) as httpx_client:
        url = test_exception_app.url + "/field-exception-units"
        response = httpx_client.post(
            url,
            json={"value": 42},
            timeout=30,
        )

        assert response.status_code == 422
        assert response.headers.get("x-fal-billable-units") == "42"


@pytest.mark.xdist_group(name="exception-app")
def test_field_exception_float_billable_units_formatting(test_exception_app: AppClient):
    """Test that float billable_units are formatted with 8 decimal places."""
    with httpx.Client(headers=_auth_headers()) as httpx_client:
        url = test_exception_app.url + "/field-exception-units"
        response = httpx_client.post(
            url,
            json={"value": 3.14159265},
            timeout=30,
        )

        assert response.status_code == 422
        assert response.headers.get("x-fal-billable-units") == "3.14159265"


@pytest.mark.xdist_group(name="exception-app")
def test_field_exception_scientific_notation_small(test_exception_app: AppClient):
    """Test that small scientific notation values are properly formatted."""
    with httpx.Client(headers=_auth_headers()) as httpx_client:
        url = test_exception_app.url + "/field-exception-units"
        response = httpx_client.post(
            url,
            json={"value": 1.23e-5},
            timeout=30,
        )

        assert response.status_code == 422
        # 1.23e-5 = 0.0000123 (float type uses .8f format)
        assert response.headers.get("x-fal-billable-units") == "0.00001230"


@pytest.mark.xdist_group(name="exception-app")
def test_field_exception_scientific_notation_large(test_exception_app: AppClient):
    """Test that large scientific notation values are properly formatted."""
    with httpx.Client(headers=_auth_headers()) as httpx_client:
        url = test_exception_app.url + "/field-exception-units"
        response = httpx_client.post(
            url,
            json={"value": 1.23e10},
            timeout=30,
        )

        assert response.status_code == 422
        # 1.23e10 = 12300000000.0 (float type uses .8f format)
        assert response.headers.get("x-fal-billable-units") == "12300000000.00000000"


@pytest.mark.xdist_group(name="exception-app")
def test_field_exception_invalid_billable_units(test_exception_app: AppClient):
    """Test that invalid billable_units (non-numeric string) raises an error."""
    with httpx.Client(headers=_auth_headers()) as httpx_client:
        url = test_exception_app.url + "/field-exception-units"
        response = httpx_client.post(
            url,
            json={"value": "not_a_number"},
            timeout=30,
        )

        # should return 500 internal server error due to ValueError when
        # converting to float
        assert response.status_code == 500


@pytest.mark.xdist_group(name="exception-app")
def test_field_exception_default_billable_units(test_exception_app: AppClient):
    """Test that when billable_units is not set (None), no header is included."""
    with httpx.Client(headers=_auth_headers()) as httpx_client:
        url = test_exception_app.url + "/field-exception"
        response = httpx_client.post(
            url,
            json={"lhs": 1, "rhs": 2},
            timeout=30,
        )

        assert response.status_code == 422
        # When billable_units is None (default), header should not be present
        assert "x-fal-billable-units" not in response.headers


def _active_runners(runners):
    active_states = {RunnerState.RUNNING, RunnerState.IDLE}
    return [runner for runner in runners if runner.state in active_states]


def submit_and_wait_for_runner(
    app: str, arguments: Optional[dict] = None, *, path: str = ""
):
    handle = apps.submit(app, arguments=arguments or {}, path=path)
    status = _wait_for_request_status(
        handle,
        (apps.InProgress, apps.Completed),
    )
    if isinstance(status, apps.Completed):
        handle.fetch_result()
    return handle


@pytest.mark.timeout(180)
def test_stop_runner(host: api.FalServerlessHost, test_sleep_app: str):
    _, _, app_alias = test_sleep_app.partition("/")
    submit_and_wait_for_runner(test_sleep_app, arguments={"wait_time": 1})

    with host._connection as client:
        runners = _wait_for_alias_runners(
            client,
            app_alias,
            lambda current: len(_active_runners(current)) == 1
            and _active_runners(current)[0].in_flight_requests == 0,
        )
        original_runner_id = _active_runners(runners)[0].runner_id

        reuse_handle = apps.submit(test_sleep_app, arguments={"wait_time": 15})
        try:
            _wait_for_request_status(reuse_handle, apps.InProgress)
            _wait_for_alias_runners(
                client,
                app_alias,
                lambda current: len(_active_runners(current)) == 1
                and _active_runners(current)[0].runner_id == original_runner_id
                and _active_runners(current)[0].in_flight_requests > 0,
            )
        finally:
            _cancel_and_wait(reuse_handle)

        _wait_for_alias_runners(
            client,
            app_alias,
            lambda current: len(_active_runners(current)) == 1
            and _active_runners(current)[0].runner_id == original_runner_id
            and _active_runners(current)[0].in_flight_requests == 0,
        )

        with pytest.raises(Exception) as exc_info:
            client.stop_runner("1234567890")
        assert "not found" in str(exc_info.value).lower()

        client.stop_runner(original_runner_id)
        _wait_for_alias_runners(
            client,
            app_alias,
            lambda current: all(
                runner.runner_id != original_runner_id
                for runner in _active_runners(current)
            ),
        )

        submit_and_wait_for_runner(test_sleep_app, arguments={"wait_time": 1})
        _wait_for_alias_runners(
            client,
            app_alias,
            lambda current: any(
                runner.runner_id != original_runner_id
                for runner in _active_runners(current)
            ),
        )


@pytest.mark.timeout(180)
def test_kill_runner(host: api.FalServerlessHost, test_sleep_app: str):
    handle = apps.submit(test_sleep_app, arguments={"wait_time": 30})
    _wait_for_request_status(handle, apps.InProgress, timeout=60)

    with host._connection as client:
        with pytest.raises(Exception) as exc_info:
            client.kill_runner("1234567890")
        assert "not found" in str(exc_info.value).lower()

        _, _, app_alias = test_sleep_app.partition("/")
        runners = _wait_for_alias_runners(
            client,
            app_alias,
            lambda current: bool(_active_runners(current)),
        )
        runner_id = _active_runners(runners)[0].runner_id

        client.kill_runner(runner_id)
        _wait_for_alias_runners(
            client,
            app_alias,
            lambda current: all(
                runner.runner_id != runner_id for runner in _active_runners(current)
            ),
        )


@pytest.mark.timeout(180)
def test_rollout_application(host: api.FalServerlessHost, test_sleep_app: str):
    handle = apps.submit(test_sleep_app, arguments={"wait_time": 30})
    _wait_for_request_status(handle, apps.InProgress, timeout=60)

    with host._connection as client:
        _, _, app_alias = test_sleep_app.partition("/")
        runners_before = _wait_for_alias_runners(
            client,
            app_alias,
            lambda current: len(_active_runners(current)) == 1,
        )
        runner_id_before = _active_runners(runners_before)[0].runner_id

        client.rollout_application(app_alias, force=True)
        runners_after = _wait_for_alias_runners(
            client,
            app_alias,
            lambda current: all(
                runner.runner_id != runner_id_before
                for runner in _active_runners(current)
            ),
            timeout=60,
        )
        runner_ids_after = {
            runner.runner_id for runner in _active_runners(runners_after)
        }

        client.rollout_application(app_alias, force=True)
        _wait_for_alias_runners(
            client,
            app_alias,
            lambda current: not runner_ids_after.intersection(
                runner.runner_id for runner in _active_runners(current)
            ),
            timeout=60,
        )


@pytest.mark.timeout(180)
def test_shell_runner(host: api.FalServerlessHost, test_sleep_app: str):
    handle = submit_and_wait_for_runner(test_sleep_app, arguments={"wait_time": 1})
    assert handle.get() == {"slept": True}

    with host._connection as client:
        _, _, app_alias = test_sleep_app.partition("/")
        runners = _wait_for_alias_runners(
            client,
            app_alias,
            lambda current: bool(_active_runners(current)),
        )
        runner_id = _active_runners(runners)[0].runner_id

        proc = subprocess.Popen(
            [sys.executable, "-m", "fal", "runners", "shell", runner_id],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            commands = b"echo 'a' > t.txt\ncat t.txt\nexit\n"
            stdout, stderr = proc.communicate(input=commands, timeout=30)
            assert b"a" in stdout, f"Expected 'a' in output, got: {stdout.decode()}"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


@pytest.mark.timeout(180)
def test_exec_runner(host: api.FalServerlessHost, test_sleep_app: str):
    handle = submit_and_wait_for_runner(test_sleep_app, arguments={"wait_time": 1})
    assert handle.get() == {"slept": True}

    with host._connection as client:
        _, _, app_alias = test_sleep_app.partition("/")
        runners = _wait_for_alias_runners(
            client,
            app_alias,
            lambda current: bool(_active_runners(current)),
        )
        runner_id = _active_runners(runners)[0].runner_id

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "fal",
                "runners",
                "exec",
                runner_id,
                "--",
                "echo",
                "hello",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = proc.communicate(timeout=30)
            assert (
                b"hello" in stdout
            ), f"Expected 'hello' in output, got: {stdout.decode()}"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


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
        with httpx.Client(headers=_auth_headers()) as httpx_client:
            url = client.url + "/add"
            resp = httpx_client.post(
                url,
                json={"lhs": 1, "rhs": 2},
                timeout=30,
            )
            assert resp.is_success
            assert resp.json()["result"] == 3


def _external_get_request_id() -> str:
    request_id = get_current_app().current_request.headers.get("x-request-id", "")
    return request_id


class AppRefOutput(BaseModel):
    from_app: str
    from_external_method: str


class AppRefApp(
    fal.App,
    keep_alive=300,
    max_concurrency=1,
    max_multiplexing=3,
):
    machine_type = "XS"

    async def setup(self):
        self.concurrent_requests = 0
        self.requests_ready = asyncio.Event()

    @fal.endpoint("/")
    async def run(self, request: Request) -> AppRefOutput:
        request_id = request.headers.get("x-request-id", "")

        self.concurrent_requests += 1
        if self.concurrent_requests == self.max_multiplexing:
            self.requests_ready.set()
        await asyncio.wait_for(self.requests_ready.wait(), timeout=30)

        return AppRefOutput(
            from_app=request_id,
            from_external_method=_external_get_request_id(),
        )


@pytest.fixture()
def test_app_ref_app(
    user: User,
    register_app,
):
    app_ref_app = wrap_app(AppRefApp)
    with register_app(app_ref_app, "app-ref") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


def test_app_ref_app_client(test_app_ref_app: str):
    handle_1 = apps.submit(test_app_ref_app, arguments={})
    handle_2 = apps.submit(test_app_ref_app, arguments={})
    handle_3 = apps.submit(test_app_ref_app, arguments={})

    result_1 = handle_1.get()
    result_2 = handle_2.get()
    result_3 = handle_3.get()

    assert result_1["from_app"] == result_1["from_external_method"]
    assert result_2["from_app"] == result_2["from_external_method"]
    assert result_3["from_app"] == result_3["from_external_method"]


@pytest.mark.timeout(180)
def test_runner_machine_type(host: api.FalServerlessHost, test_sleep_app: str):
    """Test that machine_type is populated in runner info."""
    search_start = datetime.now() - timedelta(minutes=5)
    submit_and_wait_for_runner(test_sleep_app, arguments={"wait_time": 1})

    with host._connection as client:
        _, _, app_alias = test_sleep_app.partition("/")

        runners = _wait_for_alias_runners(
            client,
            app_alias,
            lambda current: any(runner.machine_type == "XS" for runner in current),
        )
        assert any(runner.machine_type == "XS" for runner in runners)

        all_runners = _wait_until(
            lambda: client.list_runners(start_time=search_start),
            lambda current: any(runner.alias == app_alias for runner in current),
            timeout=45,
            interval=0.5,
            description=f"runner history for {app_alias}",
        )
        target_runner = next((r for r in all_runners if r.alias == app_alias), None)
        assert target_runner is not None, "Runner for test app alias not found"
        assert target_runner.machine_type == "XS"


class RequestContextOutput(BaseModel):
    request_id_from_context: Optional[str]
    endpoint_from_context: Optional[str]
    lifecycle_preference_from_context: Optional[dict]
    request_id_from_header: Optional[str]


class RequestContextInput(BaseModel):
    synchronize: bool = False


def _external_get_request_context() -> dict:
    """External function that accesses request context without request parameter."""
    current_app = get_current_app()
    if current_app is None or current_app.current_request is None:
        return {
            "request_id": None,
            "endpoint": None,
            "lifecycle_preference": None,
        }
    return {
        "request_id": current_app.current_request.request_id,
        "endpoint": current_app.current_request.endpoint,
        "lifecycle_preference": current_app.current_request.lifecycle_preference,
    }


class RequestContextApp(
    fal.App,
    keep_alive=300,
    max_concurrency=1,
    max_multiplexing=3,
):
    """App to test request context fields are properly populated."""

    machine_type = "XS"

    async def setup(self):
        self.concurrent_requests = 0
        self.requests_ready = asyncio.Event()

    @fal.endpoint("/")
    async def get_context(
        self, input: RequestContextInput, request: Request
    ) -> RequestContextOutput:
        if input.synchronize:
            self.concurrent_requests += 1
            if self.concurrent_requests == self.max_multiplexing:
                self.requests_ready.set()
            await asyncio.wait_for(self.requests_ready.wait(), timeout=30)

        context_data = _external_get_request_context()

        return RequestContextOutput(
            request_id_from_context=context_data["request_id"],
            endpoint_from_context=context_data["endpoint"],
            lifecycle_preference_from_context=context_data["lifecycle_preference"],
            request_id_from_header=request.headers.get("x-fal-request-id"),
        )


@pytest.fixture(scope="module")
def test_request_context_app(
    user: User,
    register_app,
):
    request_context_app = wrap_app(RequestContextApp)
    with register_app(request_context_app, "request-context") as (app_alias, _):
        yield f"{user.username}/{app_alias}"


@pytest.mark.xdist_group(name="request-context-app")
def test_request_context_fields_populated(test_request_context_app: str):
    """Test that request context fields are properly populated."""

    result = apps.run(
        test_request_context_app,
        arguments={"synchronize": False},
    )

    assert result["request_id_from_context"] is not None
    assert result["endpoint_from_context"] is not None
    assert result["lifecycle_preference_from_context"] is not None

    assert result["request_id_from_context"] == result["request_id_from_header"]


@pytest.mark.xdist_group(name="request-context-app")
def test_request_context_isolation_with_multiplexing(test_request_context_app: str):
    """Test that request context is properly isolated between concurrent requests.

    With multiplexing enabled (max_multiplexing=3), each request should have
    its own isolated context via ContextVar, ensuring request_id from context
    matches the request_id from headers for each individual request.
    """

    arguments = {"synchronize": True}
    handle_1 = apps.submit(test_request_context_app, arguments=arguments)
    handle_2 = apps.submit(test_request_context_app, arguments=arguments)
    handle_3 = apps.submit(test_request_context_app, arguments=arguments)

    # Get results
    result_1 = handle_1.get()
    result_2 = handle_2.get()
    result_3 = handle_3.get()

    # Each request's context should match its own header, proving isolation
    assert result_1["request_id_from_context"] == result_1["request_id_from_header"]
    assert result_2["request_id_from_context"] == result_2["request_id_from_header"]
    assert result_3["request_id_from_context"] == result_3["request_id_from_header"]

    # All request IDs should be different (unique requests)
    request_ids = {
        result_1["request_id_from_context"],
        result_2["request_id_from_context"],
        result_3["request_id_from_context"],
    }
    assert len(request_ids) == 3, "Each request should have a unique request_id"
