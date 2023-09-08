import fal
import pytest
from fal import apps
from pydantic import BaseModel


@pytest.fixture(scope="module")
def test_app() -> str:
    # Create a temporary app, register it, and return the ID of it.

    import time

    from fal.cli import _get_user_id

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
    )
    def addition_app(input: Input) -> Output:
        print("starting...")
        for _ in range(input.wait_time):
            print("sleeping...")
            time.sleep(1)

        return Output(result=input.lhs + input.rhs)

    app_alias = addition_app.host.register(
        func=addition_app.func,
        options=addition_app.options,
        max_concurrency=1,
    )
    user_id = _get_user_id()
    yield f"{user_id}-{app_alias}"


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
    for event in request_handle.iter_events():
        assert isinstance(event, (apps.Queued, apps.InProgress))
        if isinstance(event, apps.InProgress) and event.logs:
            logs = [log["message"] for log in event.logs]
            assert "sleeping..." in logs
        elif isinstance(event, apps.Queued):
            assert event.position == 0

    assert isinstance(request_handle.status(), apps.Completed)

    # It is safe to use fetch_result when we know for a fact the request itself
    # is completed.
    result = request_handle.fetch_result()

    # .get() can still be used and will return the same value
    result_alternative = request_handle.get()
    assert result == result_alternative
    assert result == {"result": 5}
