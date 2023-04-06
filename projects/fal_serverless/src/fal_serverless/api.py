from __future__ import annotations

import inspect
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial, wraps
from typing import Any, Callable, ClassVar, Dict, Generic, TypeVar, cast

import dill
import grpc
import isolate
import yaml
from fal_serverless.flags import bool_envvar
from fal_serverless.logging.isolate import IsolateLogPrinter
from fal_serverless.sdk import (
    FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    Credentials,
    FalServerlessClient,
    FalServerlessConnection,
    HostedRunState,
    MachineRequirements,
    _get_agent_credentials,
    get_default_credentials,
)
from isolate.backends.common import active_python
from isolate.backends.settings import DEFAULT_SETTINGS
from isolate.connections import PythonIPC

dill.settings["recurse"] = True

ReturnT = TypeVar("ReturnT")
BasicConfig = Dict[str, Any]
_UNSET = object()


@dataclass
class FalServerlessError(Exception):
    message: str


@dataclass
class InternalFalServerlessError(Exception):
    ...


@dataclass
class Host:
    """The physical environment where the isolated code
    is executed."""

    _SUPPORTED_KEYS: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def parse_key(cls, key: str, value: Any) -> tuple[Any, Any]:
        if key == "env_yml":
            # Conda environment definition should be parsed before sending to serverless
            with open(value) as f:
                return "env_dict", yaml.safe_load(f)
        else:
            return key, value

    @classmethod
    def parse_options(cls, **config: Any) -> Options:
        """Split the given set of options into host and
        environment options."""

        options = Options()
        for key, value in config.items():
            key, value = cls.parse_key(key, value)
            if key in cls._SUPPORTED_KEYS:
                options.host[key] = value
            elif key == "serve" or key == "exposed_port":
                options.gateway[key] = value
            else:
                options.environment[key] = value

        if options.gateway.get("serve"):
            options.add_requirements(["flask", "flask-cors"])

        return options

    def run(
        self,
        func: Callable[..., ReturnT],
        options: Options,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ReturnT:
        """Run the given function in the isolated environment."""
        raise NotImplementedError


def cached(func: Callable[..., ReturnT]) -> Callable[..., ReturnT]:
    """Cache the result of the given function in-memory."""
    import hashlib

    try:
        source_code = inspect.getsource(func).encode("utf-8")
    except OSError:
        # TODO: explain the reason for this (e.g. we don't know how to
        # check if you sent us the same function twice).
        print(f"[warning] Function {func.__name__} can not be cached...")
        return func

    cache_key = hashlib.sha256(source_code).hexdigest()

    @wraps(func)
    def wrapper(*args, **kwargs) -> ReturnT:
        from functools import lru_cache

        import isolate

        if not hasattr(isolate, "__cached_functions__"):
            isolate.__cached_functions__ = {}

        if cache_key not in isolate.__cached_functions__:
            isolate.__cached_functions__[cache_key] = lru_cache(maxsize=None)(func)

        return isolate.__cached_functions__[cache_key](*args, **kwargs)

    return wrapper


def _execution_controller(
    func: Callable[..., ReturnT],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Callable[..., ReturnT]:
    """Handle the execution of the given user function."""

    @wraps(func)
    def wrapper(*remote_args: Any, **remote_kwargs: Any) -> ReturnT:
        return func(*remote_args, *args, **remote_kwargs, **kwargs)

    return wrapper


@dataclass
class LocalHost(Host):
    # The environment which provides the default set of
    # packages for isolate agent to run.
    _AGENT_ENVIRONMENT = isolate.prepare_environment(
        "virtualenv",
        requirements=[f"dill=={dill.__version__}"],
    )

    def run(
        self,
        func: Callable[..., ReturnT],
        options: Options,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ReturnT:
        settings = replace(DEFAULT_SETTINGS, serialization_method="dill")
        environment = isolate.prepare_environment(
            **options.environment,
            context=settings,
        )
        with PythonIPC(
            environment,
            environment.create(),
            extra_inheritance_paths=[self._AGENT_ENVIRONMENT.create()],
        ) as connection:
            executable = partial(func, *args, **kwargs)
            return connection.run(executable)


FAL_SERVERLESS_DEFAULT_URL = os.getenv("FAL_HOST", "api.alpha.fal.ai:443")
FAL_SERVERLESS_DEFAULT_MACHINE_TYPE = "XS"


import threading


def _handle_grpc_error():
    def decorator(fn):
        @wraps(fn)
        def handler(*args, **kwargs):
            """
            Wraps grpc errors as fal Serverless Errors.
            """
            max_retries = 3
            max_wait_time = 1
            for i in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except grpc.RpcError as e:
                    if e.code() == grpc.StatusCode.UNAVAILABLE:
                        print("Could not reach fal Serverless host. Trying again.")
                        time.sleep(max_wait_time)

                        # Reached max retries
                        if i + 1 == max_retries:
                            raise FalServerlessError(
                                "Could not reach fal Serverless host. "
                                "This is most likely a transient problem. "
                                "Please, try again."
                            )
                    elif e.details().endswith("died with <Signals.SIGKILL: 9>.`."):
                        raise FalServerlessError(
                            "Isolated function crashed. "
                            "This is likely due to resource overflow. "
                            "You can try again by setting a bigger `machine_type`"
                        )
                    else:
                        raise FalServerlessError(e.details())

        return handler

    return decorator


# TODO: Should we build all these in fal/dbt-fal packages instead?
@dataclass
class FalServerlessHost(Host):
    _SUPPORTED_KEYS = frozenset({"machine_type", "keep_alive", "setup_function"})

    url: str = FAL_SERVERLESS_DEFAULT_URL
    credentials: Credentials = field(default_factory=get_default_credentials)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    _log_printer = IsolateLogPrinter(debug=bool_envvar("DEBUG"))

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.credentials = _get_agent_credentials(self.credentials)

    @property
    def _connection(self) -> FalServerlessConnection:
        with self._lock:
            client = FalServerlessClient(self.url, self.credentials)
            return client.connect()

    @_handle_grpc_error()
    def register(
        self,
        func: Callable[..., ReturnT],
        options: Options,
        application_name: str | None = None,
    ) -> str | None:
        environment_options = options.environment.copy()
        environment_options.setdefault("python_version", active_python())
        environments = [self._connection.define_environment(**environment_options)]

        machine_type = options.host.get(
            "machine_type", FAL_SERVERLESS_DEFAULT_MACHINE_TYPE
        )
        keep_alive = options.host.get("keep_alive", FAL_SERVERLESS_DEFAULT_KEEP_ALIVE)

        machine_requirements = MachineRequirements(
            machine_type=machine_type, keep_alive=keep_alive
        )

        partial_func = _execution_controller(func, tuple(), {})

        for partial_result in self._connection.register(
            partial_func,
            environments,
            application_name=application_name,
            machine_requirements=machine_requirements,
        ):
            for log in partial_result.logs:
                self._log_printer.print(log)

            if partial_result.result:
                return partial_result.result.application_id

    @_handle_grpc_error()
    def schedule(
        self, func: Callable[..., ReturnT], cron: str, options: Options
    ) -> str | None:
        application_id = self.register(func, options)
        cron_id = self._connection.schedule_cronjob(application_id, cron)
        return cron_id

    @_handle_grpc_error()
    def run(
        self,
        func: Callable[..., ReturnT],
        options: Options,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ReturnT:
        environment_options = options.environment.copy()
        environment_options.setdefault("python_version", active_python())
        environments = [self._connection.define_environment(**environment_options)]

        machine_type = options.host.get(
            "machine_type", FAL_SERVERLESS_DEFAULT_MACHINE_TYPE
        )
        keep_alive = options.host.get("keep_alive", FAL_SERVERLESS_DEFAULT_KEEP_ALIVE)
        setup_function = options.host.get("setup_function", None)

        machine_requirements = MachineRequirements(
            machine_type=machine_type, keep_alive=keep_alive
        )

        return_value = _UNSET
        # Allow isolate provided arguments (such as setup function) to take
        # precedence over the ones provided by the user.
        partial_func = _execution_controller(func, args, kwargs)
        for partial_result in self._connection.run(
            partial_func,
            environments,
            machine_requirements=machine_requirements,
            setup_function=setup_function,
        ):
            for log in partial_result.logs:
                self._log_printer.print(log)

            if partial_result.status.state is not HostedRunState.IN_PROGRESS:
                state = partial_result.status.state
                if state is HostedRunState.INTERNAL_FAILURE:
                    raise InternalFalServerlessError(
                        "An internal failure occurred while performing this run."
                    )
                elif state is HostedRunState.SUCCESS:
                    return_value = partial_result.result
                else:
                    raise NotImplementedError("Unknown state: ", state)

        if return_value is _UNSET:
            raise InternalFalServerlessError(
                "The input function did not return any value."
            )

        return cast(ReturnT, return_value)


@dataclass
class Options:
    host: BasicConfig = field(default_factory=dict)
    environment: BasicConfig = field(default_factory=dict)
    gateway: BasicConfig = field(default_factory=dict)

    def add_requirements(self, requirements: list[str]):
        kind = self.environment["kind"]
        if kind == "virtualenv":
            pip_requirements = self.environment.setdefault("requirements", [])
        elif kind == "conda":
            pip_requirements = self.environment.setdefault("pip", [])
        else:
            raise FalServerlessError(
                "Only conda and virtualenv is supported as environment options"
            )
        pip_requirements.extend(requirements)


_DEFAULT_HOST = FalServerlessHost()


def isolated(
    kind: str = "virtualenv",
    *,
    host: Host = _DEFAULT_HOST,
    **config: Any,
) -> Callable[[Callable[..., ReturnT]], IsolatedFunction[ReturnT]]:
    options = host.parse_options(kind=kind, **config)

    def wrapper(func: Callable[..., ReturnT]) -> IsolatedFunction[ReturnT]:
        # wrap it with flask if the serve option is set
        func = templated_flask(func) if options.gateway.get("serve") else func
        proxy = IsolatedFunction(
            host=host,
            func=func,
            options=options,
        )
        return wraps(func)(proxy)

    return wrapper


def templated_flask(func: Callable[..., ReturnT]) -> Callable[..., ReturnT]:
    param_names = inspect.signature(func).parameters.keys()

    def templated_flask_wrapper() -> Any:
        from flask import Flask, jsonify, request
        from flask_cors import CORS

        app = Flask("fal")
        cors = CORS(app, resources={r"/*": {"origins": "*"}})

        @app.route("/", methods=["POST"])
        def flask():
            params = [request.get_json().get(param) for param in param_names]
            return jsonify({"result": func(*params)})

        app.run(host="0.0.0.0", port=8080)

    return templated_flask_wrapper


@dataclass
class IsolatedFunction(Generic[ReturnT]):
    host: Host
    func: Callable[..., ReturnT]
    options: Options
    executor: ThreadPoolExecutor = field(default_factory=ThreadPoolExecutor)

    def __getstate__(self) -> dict[str, Any]:
        # Ensure that the executor is not pickled.
        state = self.__dict__.copy()
        del state["executor"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "executor"):
            self.executor = ThreadPoolExecutor()

    def submit(self, *args: Any, **kwargs: Any) -> Future[ReturnT]:
        # TODO: This should probably live inside each host since they can
        # have more optimized Future implementations (e.g. instead of real
        # threads, they can use state observers and detached runs).

        future = self.executor.submit(
            self.host.run,
            func=self.func,  # type: ignore
            options=self.options,
            args=args,
            kwargs=kwargs,
        )
        return cast(Future[ReturnT], future)

    def __call__(self, *args, **kwargs) -> ReturnT:
        return self.host.run(
            self.func,
            self.options,
            args=args,
            kwargs=kwargs,
        )

    def on(self, host: Host | None = None, **config: Any) -> IsolatedFunction[ReturnT]:
        host = host or self.host
        if isinstance(host, type(self.host)):
            previous_host_options = self.options.host
        else:
            previous_host_options = {}

        # The order of the options is important here (the latter
        # options override the former ones).
        host_options = {**previous_host_options, **config}
        new_options = host.parse_options(**self.options.environment, **host_options)
        return replace(
            self,
            host=host,
            options=new_options,
        )
