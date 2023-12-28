from typing import Generator

import fal
import fal.api as api
import pytest
from fal import apps
from fal.rest_client import REST_CLIENT
import time
from pydantic import BaseModel

from openapi_fal_rest.api.applications import app_metadata


class Input(BaseModel):
    lhs: int
    rhs: int
    wait_time: int = 0


class Output(BaseModel):
    result: int


@fal.function(
    keep_alive=60,
    machine_type="S",
    serve=True,
    max_concurrency=1,
)
def addition_app(input: Input) -> Output:
    print("starting...")
    for _ in range(input.wait_time):
        print("sleeping...")
        time.sleep(1)

    return Output(result=input.lhs + input.rhs)


@fal.function(
    keep_alive=0,
    requirements=["fastapi", "uvicorn", "pydantic==1.10.12"],
    machine_type="S",
    max_concurrency=1,
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


@pytest.fixture(scope="module")
def aliased_app() -> Generator[tuple[str, str], None, None]:
    # Create a temporary app, register it, and return the ID of it.

    import uuid

    app_alias = str(uuid.uuid4()).replace("-", "")[:10]
    app_revision = addition_app.host.register(
        func=addition_app.func,
        options=addition_app.options,
        # random enough
        application_name=app_alias,
        application_auth_mode="private",
    )
    yield app_revision, app_alias  # type: ignore


@pytest.fixture(scope="module")
def test_app():
    # Create a temporary app, register it, and return the ID of it.

    from fal.cli import _get_user_id

    app_revision = addition_app.host.register(
        func=addition_app.func,
        options=addition_app.options,
    )
    user_id = _get_user_id()
    yield f"{user_id}-{app_revision}"


@pytest.fixture(scope="module")
def test_fastapi_app():
    # Create a temporary app, register it, and return the ID of it.

    from fal.cli import _get_user_id

    app_revision = calculator_app.host.register(
        func=calculator_app.func,
        options=calculator_app.options,
    )
    user_id = _get_user_id()
    yield f"{user_id}-{app_revision}"


def test_app_client(test_app: str):
    response = apps.run(test_app, arguments={"lhs": 1, "rhs": 2})
    assert response["result"] == 3

    response = apps.run(test_app, arguments={"lhs": 2, "rhs": 3, "wait_time": 1})
    assert response["result"] == 5


def test_app_client_async(test_app: str):
    request_handle = apps.submit(test_app, arguments={"lhs": 1, "rhs": 2})
    assert request_handle.get() == {"result": 3}

    request_handle = apps.submit(
        test_app, arguments={"lhs": 2, "rhs": 3, "wait_time": 5}
    )

    for event in request_handle.iter_events(logs=True):
        assert isinstance(event, (apps.Queued, apps.InProgress))
        if isinstance(event, apps.InProgress) and event.logs:
            logs = [log["message"] for log in event.logs]
            assert "sleeping..." in logs
        elif isinstance(event, apps.Queued):
            assert event.position == 0

    status = request_handle.status(logs=True)
    assert isinstance(status, apps.Completed)
    assert status.logs, "Logs missing from Completed status"

    # It is safe to use fetch_result when we know for a fact the request itself
    # is completed.
    result = request_handle.fetch_result()

    # .get() can still be used and will return the same value
    result_alternative = request_handle.get()
    assert result == result_alternative
    assert result == {"result": 5}


@pytest.mark.parametrize("app_name", ["test_app", "test_fastapi_app"])
def test_app_openapi_spec_metadata(app_name: str, request: pytest.FixtureRequest):
    app: str = request.getfixturevalue(app_name)
    user_id, _, app_id = app.partition("-")
    res = app_metadata.sync_detailed(
        app_alias_or_id=app_id, app_user_id=user_id, client=REST_CLIENT
    )

    assert res.status_code == 200, f"Failed to fetch metadata for app {app}"
    assert res.parsed, f"Failed to parse metadata for app {app}"

    metadata = res.parsed.to_dict()
    assert "openapi" in metadata, f"openapi key missing from metadata {metadata}"
    openapi_spec: dict = metadata["openapi"]
    for key in ["openapi", "info", "paths", "components"]:
        assert key in openapi_spec, f"{key} key missing from openapi {openapi_spec}"


def test_app_update_app(aliased_app: tuple[str, str]):
    app_revision, app_alias = aliased_app

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
        new_max_concurrency = new_max_concurrency + 1
        res = client.update_application(
            application_name=app_alias,
            max_concurrency=new_max_concurrency,
        )
        assert res.alias == app_alias
        assert res.keep_alive == new_keep_alive
        assert res.max_concurrency == new_max_concurrency
        assert res.max_multiplexing == new_max_multiplexing


def test_app_set_delete_alias(aliased_app: tuple[str, str]):
    app_revision, app_alias = aliased_app

    host: api.FalServerlessHost = addition_app.host  # type: ignore

    with host._connection as client:
        # Get the registered values
        res = client.list_aliases()
        found = next(filter(lambda alias: alias.alias == app_alias, res), None)
        assert found, f"Could not find app {app_alias} in {res}"
        assert found.revision == app_revision
        assert found.auth_mode == "private"

    new_app_alias = f"{app_alias}-new"
    with host._connection as client:
        # Get the registered values
        res = client.create_alias(new_app_alias, app_revision, "public")

    with host._connection as client:
        # Get the registered values
        res = client.list_aliases()
        found = next(filter(lambda alias: alias.alias == new_app_alias, res), None)
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
