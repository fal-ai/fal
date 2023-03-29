from __future__ import annotations

import pytest
from fal_serverless import FalServerlessHost, FalServerlessKeyCredentials, local
from fal_serverless.api import FalServerlessError


def test_isolated(isolated_client):
    @isolated_client("virtualenv", requirements=["pyjokes==0.5.0"])
    def get_pyjokes_version():
        import pyjokes

        return pyjokes.__version__

    result = get_pyjokes_version()
    assert result == "0.5.0"

    @isolated_client("virtualenv")
    def get_hostname() -> str:
        import socket

        return socket.gethostname()

    first = get_hostname()
    assert first.startswith("worker")

    get_hostname_local = get_hostname.on(local)
    second = get_hostname_local()
    assert not second.startswith("worker-")

    get_hostname_m = get_hostname.on(machine_type="M")
    third = get_hostname_m()
    assert third.startswith("worker")
    assert third != first

    # The machine_type should be dropped when using local
    get_hostname_m_local = get_hostname_m.on(local)
    fourth = get_hostname_m_local()
    assert not fourth.startswith("worker-")


def test_isolate_setup_funcs(isolated_client):
    def setup_function():
        import math

        return math.pi

    @isolated_client(setup_function=setup_function)
    def is_tau(setup, by_factor) -> str:
        import math

        return setup * by_factor == math.tau

    assert is_tau(2)
    assert not is_tau(by_factor=3)


def test_isolate_setup_func_order(isolated_client):
    def setup_function():
        return "one "

    @isolated_client(setup_function=setup_function)
    def one_and(setup, num) -> str:
        return setup + num

    assert one_and("two") == "one two"
    assert one_and("three") == "one three"


def test_isolate_error_handling(isolated_client):
    requirements = ["pyjokes", "requests"]

    def setup():
        print("hello")

    @isolated_client(requirements=requirements, keep_alive=20, setup=setup)
    def raises_value_error():
        import pyjokes

        return pyjokes.get_joke()

    with pytest.raises(ValueError):
        raises_value_error()

    creds = FalServerlessKeyCredentials(key_id="fake", key_secret="fake")
    host = FalServerlessHost(url="not.there", credentials=creds)

    @isolated_client(host=host)
    def raises_grpc_error():
        return True

    with pytest.raises(FalServerlessError):
        raises_grpc_error()
