from __future__ import annotations

import inspect
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from functools import wraps
from os import PathLike
from queue import Queue
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    Literal,
    NamedTuple,
    TypeVar,
    cast,
    overload,
)

import cloudpickle
import grpc
import isolate
import tblib
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi import __version__ as fastapi_version
from isolate.backends.common import active_python
from isolate.backends.settings import DEFAULT_SETTINGS
from isolate.connections import PythonIPC
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from pydantic import __version__ as pydantic_version
from typing_extensions import Concatenate, ParamSpec

import fal.flags as flags
from fal._serialization import include_modules_from, patch_pickle
from fal.container import ContainerImage
from fal.exceptions import (
    AppException,
    CUDAOutOfMemoryException,
    FalServerlessException,
    FieldException,
)
from fal.exceptions._cuda import _is_cuda_oom_exception
from fal.logging.isolate import IsolateLogPrinter
from fal.sdk import (
    FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    FAL_SERVERLESS_DEFAULT_MAX_MULTIPLEXING,
    FAL_SERVERLESS_DEFAULT_MIN_CONCURRENCY,
    Credentials,
    FalServerlessClient,
    FalServerlessConnection,
    HostedRunState,
    MachineRequirements,
    get_agent_credentials,
    get_default_credentials,
)

ArgsT = ParamSpec("ArgsT")
ReturnT = TypeVar("ReturnT", covariant=True)  # noqa: PLC0105

BasicConfig = Dict[str, Any]
_UNSET = object()

SERVE_REQUIREMENTS = [
    f"fastapi=={fastapi_version}",
    f"pydantic=={pydantic_version}",
    "uvicorn",
    "starlette_exporter",
    "structlog",
]


THREAD_POOL = ThreadPoolExecutor()


@dataclass
class FalServerlessError(FalServerlessException):
    message: str


@dataclass
class InternalFalServerlessError(FalServerlessException):
    message: str


@dataclass
class FalMissingDependencyError(FalServerlessError): ...


@dataclass
class SpawnInfo:
    future: Future | None = None
    logs: Queue = field(default_factory=Queue)
    _url_ready: threading.Event = field(default_factory=threading.Event)
    _url: str | None = None
    stream: Any = None

    @property
    def return_value(self):
        if self.future is None:
            raise ValueError
        return self.future.result()

    @property
    def url(self):
        self._url_ready.wait()
        return self._url

    @url.setter
    def url(self, value):
        self._url_ready.set()
        self._url = value


@dataclass
class Host(Generic[ArgsT, ReturnT]):
    """The physical environment where the isolated code
    is executed."""

    _SUPPORTED_KEYS: ClassVar[frozenset[str]] = frozenset()
    _GATEWAY_KEYS: ClassVar[frozenset[str]] = frozenset({"serve", "exposed_port"})

    def __post_init__(self):
        assert not self._SUPPORTED_KEYS.intersection(
            self._GATEWAY_KEYS
        ), "Gateway keys and host keys should not overlap"

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
        for item in config.items():
            key, value = cls.parse_key(*item)
            if key in cls._SUPPORTED_KEYS:
                options.host[key] = value
            elif key in cls._GATEWAY_KEYS:
                options.gateway[key] = value
            else:
                options.environment[key] = value

        if options.gateway.get("serve"):
            options.add_requirements(SERVE_REQUIREMENTS)

        return options

    def register(
        self,
        func: Callable[ArgsT, ReturnT],
        options: Options,
        application_name: str | None = None,
        application_auth_mode: Literal["public", "shared", "private"] | None = None,
        metadata: dict[str, Any] | None = None,
        scale: bool = True,
    ) -> str | None:
        """Register the given function on the host for API call execution."""
        raise NotImplementedError

    def run(
        self,
        func: Callable[ArgsT, ReturnT],
        options: Options,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ReturnT:
        """Run the given function in the isolated environment."""
        raise NotImplementedError

    def spawn(
        self,
        func: Callable[ArgsT, ReturnT],
        options: Options,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> SpawnInfo:
        raise NotImplementedError


def cached(func: Callable[ArgsT, ReturnT]) -> Callable[ArgsT, ReturnT]:
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
    def wrapper(
        *args: ArgsT.args,
        **kwargs: ArgsT.kwargs,
    ) -> ReturnT:
        from functools import lru_cache

        # HACK: Using the isolate module as a global cache.
        import isolate

        if not hasattr(isolate, "__cached_functions__"):
            isolate.__cached_functions__ = {}

        if cache_key not in isolate.__cached_functions__:
            isolate.__cached_functions__[cache_key] = lru_cache(maxsize=None)(func)

        return isolate.__cached_functions__[cache_key](*args, **kwargs)

    return wrapper


class UserFunctionException(FalServerlessException):
    pass


def _prepare_partial_func(
    func: Callable[ArgsT, ReturnT],
    *args: ArgsT.args,
    **kwargs: ArgsT.kwargs,
) -> Callable[ArgsT, ReturnT]:
    """Prepare the given function for execution on isolate workers."""

    @wraps(func)
    def wrapper(*remote_args: ArgsT.args, **remote_kwargs: ArgsT.kwargs) -> ReturnT:
        try:
            result = func(*remote_args, *args, **remote_kwargs, **kwargs)
        except FalServerlessException:
            raise
        except Exception as exc:
            tb = exc.__traceback__
            if tb is not None and tb.tb_next is not None:
                # remove our wrapper from user's traceback
                tb = tb.tb_next
            raise UserFunctionException(
                f"Uncaught user function exception: {str(exc)}"
            ) from exc.with_traceback(tb)
        finally:
            with suppress(Exception):
                patch_pickle()
        return result

    return wrapper


@dataclass
class LocalHost(Host):
    # The environment which provides the default set of
    # packages for isolate agent to run.
    _AGENT_ENVIRONMENT = isolate.prepare_environment(
        "virtualenv",
        requirements=[
            f"cloudpickle=={cloudpickle.__version__}",
            f"tblib=={tblib.__version__}",
        ],
    )
    _log_printer = IsolateLogPrinter(debug=flags.DEBUG)

    def run(
        self,
        func: Callable[..., ReturnT],
        options: Options,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ReturnT:
        settings = replace(
            DEFAULT_SETTINGS,
            serialization_method="cloudpickle",
            log_hook=self._log_printer.print,
        )
        environment = isolate.prepare_environment(
            **options.environment,
            context=settings,
        )
        with PythonIPC(
            environment,
            environment.create(),
            extra_inheritance_paths=[self._AGENT_ENVIRONMENT.create()],
        ) as connection:
            executable = _prepare_partial_func(func, *args, **kwargs)
            return connection.run(executable)


FAL_SERVERLESS_DEFAULT_URL = flags.GRPC_HOST
FAL_SERVERLESS_DEFAULT_MACHINE_TYPE = "XS"


def _handle_grpc_error():
    def decorator(fn):
        @wraps(fn)
        def handler(*args, **kwargs):
            """
            Wraps grpc errors as fal Serverless Errors.
            """
            try:
                return fn(*args, **kwargs)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    raise FalServerlessError(
                        "Could not reach fal host. "
                        "This is most likely a transient problem. "
                        "Please, try again."
                    )
                elif e.details().endswith("died with <Signals.SIGKILL: 9>.`."):
                    raise FalServerlessError(
                        "Isolated function crashed. "
                        "This is likely due to resource overflow. "
                        "You can try again by setting a bigger `machine_type`"
                    )
                elif e.code() == grpc.StatusCode.INVALID_ARGUMENT and (
                    "The function function could not be deserialized" in e.details()
                ):
                    raise FalMissingDependencyError(e.details()) from None
                else:
                    raise FalServerlessError(e.details())

        return handler

    return decorator


def find_missing_dependencies(
    func: Callable, env: dict
) -> Iterator[tuple[str, list[str]]]:
    import dill

    if env["kind"] != "virtualenv":
        return

    used_modules = defaultdict(list)
    scope = {**dill.detect.globalvars(func, recurse=True), **dill.detect.freevars(func)}  # type: ignore

    for name, obj in scope.items():
        if isinstance(obj, IsolatedFunction):
            used_modules["fal"].append(name)
            continue

        module = inspect.getmodule(obj)
        if module is None:
            continue

        possible_package = getattr(module, "__package__", None)
        if possible_package:
            pkg_name, *_ = possible_package.split(".")  # type: ignore
        else:
            pkg_name = module.__name__

        used_modules[canonicalize_name(pkg_name)].append(name)  # type: ignore

    raw_requirements = env.get("requirements", [])
    specified_requirements = set()
    for raw_requirement in raw_requirements:
        try:
            requirement = Requirement(raw_requirement)
        except ValueError:
            # For git+ dependencies, we can't parse the canonical name
            # so we'll just skip them.
            continue
        specified_requirements.add(canonicalize_name(requirement.name))

    for module_name, used_names in used_modules.items():
        if module_name in specified_requirements:
            continue
        yield module_name, used_names


# TODO: Should we build all these in fal/dbt-fal packages instead?
@dataclass
class FalServerlessHost(Host):
    _SUPPORTED_KEYS = frozenset(
        {
            "machine_type",
            "machine_types",
            "num_gpus",
            "keep_alive",
            "max_concurrency",
            "min_concurrency",
            "max_multiplexing",
            "setup_function",
            "metadata",
            "request_timeout",
            "_base_image",
            "_scheduler",
            "_scheduler_options",
        }
    )

    url: str = FAL_SERVERLESS_DEFAULT_URL
    credentials: Credentials = field(default_factory=get_default_credentials)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    _log_printer = IsolateLogPrinter(debug=flags.DEBUG)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.credentials = get_agent_credentials(self.credentials)

    @property
    def _connection(self) -> FalServerlessConnection:
        with self._lock:
            client = FalServerlessClient(self.url, self.credentials)
            return client.connect()

    @_handle_grpc_error()
    def register(
        self,
        func: Callable[ArgsT, ReturnT],
        options: Options,
        application_name: str | None = None,
        application_auth_mode: Literal["public", "shared", "private"] | None = None,
        metadata: dict[str, Any] | None = None,
        deployment_strategy: Literal["recreate", "rolling"] = "recreate",
        scale: bool = True,
    ) -> str | None:
        environment_options = options.environment.copy()
        environment_options.setdefault("python_version", active_python())
        environments = [self._connection.define_environment(**environment_options)]

        machine_type: list[str] | str = options.host.get(
            "machine_type", FAL_SERVERLESS_DEFAULT_MACHINE_TYPE
        )
        keep_alive = options.host.get("keep_alive", FAL_SERVERLESS_DEFAULT_KEEP_ALIVE)
        base_image = options.host.get("_base_image", None)
        scheduler = options.host.get("_scheduler", None)
        scheduler_options = options.host.get("_scheduler_options", None)
        max_concurrency = options.host.get("max_concurrency")
        min_concurrency = options.host.get("min_concurrency")
        max_multiplexing = options.host.get("max_multiplexing")
        exposed_port = options.get_exposed_port()
        request_timeout = options.host.get("request_timeout")
        machine_requirements = MachineRequirements(
            machine_types=machine_type,  # type: ignore
            num_gpus=options.host.get("num_gpus"),
            keep_alive=keep_alive,
            base_image=base_image,
            exposed_port=exposed_port,
            scheduler=scheduler,
            scheduler_options=scheduler_options,
            max_multiplexing=max_multiplexing,
            max_concurrency=max_concurrency,
            min_concurrency=min_concurrency,
            request_timeout=request_timeout,
        )

        partial_func = _prepare_partial_func(func)

        if metadata is None:
            metadata = {}

        # TODO: let the user send more metadata than just openapi
        if isinstance(func, ServeWrapper):
            # Assigning in a separate property leaving a place for the user
            # to add more metadata in the future
            try:
                metadata["openapi"] = func.openapi()
            except Exception as e:
                print(
                    f"[warning] Failed to generate OpenAPI metadata for function: {e}"
                )

        for partial_result in self._connection.register(
            partial_func,
            environments,
            application_name=application_name,
            application_auth_mode=application_auth_mode,
            machine_requirements=machine_requirements,
            metadata=metadata,
            deployment_strategy=deployment_strategy,
            scale=scale,
        ):
            for log in partial_result.logs:
                self._log_printer.print(log)

            if partial_result.result:
                return partial_result.result.application_id

        return None

    @_handle_grpc_error()
    def _run(
        self,
        func: Callable[..., ReturnT],
        options: Options,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result_handler: Callable[..., None],
    ) -> ReturnT:
        environment_options = options.environment.copy()
        environment_options.setdefault("python_version", active_python())
        environments = [self._connection.define_environment(**environment_options)]

        machine_type: list[str] | str = options.host.get(
            "machine_type", FAL_SERVERLESS_DEFAULT_MACHINE_TYPE
        )
        keep_alive = options.host.get("keep_alive", FAL_SERVERLESS_DEFAULT_KEEP_ALIVE)
        max_concurrency = options.host.get("max_concurrency")
        min_concurrency = options.host.get("min_concurrency")
        max_multiplexing = options.host.get("max_multiplexing")
        base_image = options.host.get("_base_image", None)
        scheduler = options.host.get("_scheduler", None)
        scheduler_options = options.host.get("_scheduler_options", None)
        exposed_port = options.get_exposed_port()
        setup_function = options.host.get("setup_function", None)
        request_timeout = options.host.get("request_timeout")

        machine_requirements = MachineRequirements(
            machine_types=machine_type,  # type: ignore
            num_gpus=options.host.get("num_gpus"),
            keep_alive=keep_alive,
            base_image=base_image,
            exposed_port=exposed_port,
            scheduler=scheduler,
            scheduler_options=scheduler_options,
            max_multiplexing=max_multiplexing,
            max_concurrency=max_concurrency,
            min_concurrency=min_concurrency,
            request_timeout=request_timeout,
        )

        return_value = _UNSET
        # Allow isolate provided arguments (such as setup function) to take
        # precedence over the ones provided by the user.
        partial_func = _prepare_partial_func(func, *args, **kwargs)
        for partial_result in self._connection.run(
            partial_func,
            environments,
            machine_requirements=machine_requirements,
            setup_function=setup_function,
        ):
            result_handler(partial_result)

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

    def run(
        self,
        func: Callable[..., ReturnT],
        options: Options,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ReturnT:
        def result_handler(partial_result):
            for log in partial_result.logs:
                self._log_printer.print(log)

        return self._run(func, options, args, kwargs, result_handler=result_handler)

    def spawn(
        self,
        func: Callable[..., ReturnT],
        options: Options,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> SpawnInfo:
        ret = SpawnInfo()

        def result_handler(partial_result):
            ret.stream = partial_result.stream
            for log in partial_result.logs:
                if "Access your exposed service at" in log.message:
                    ret.url = log.message.rsplit()[-1]
                ret.logs.put(log)

        THREAD_POOL.submit(
            self._run,
            func,
            options,
            args,
            kwargs,
            result_handler=result_handler,
        )

        return ret


@dataclass
class Options:
    host: BasicConfig = field(default_factory=dict)
    environment: BasicConfig = field(default_factory=dict)
    gateway: BasicConfig = field(default_factory=dict)

    def add_requirements(self, requirements: list[str]):
        kind = self.environment["kind"]
        if kind in ["virtualenv", "container"]:
            pip_requirements = self.environment.setdefault("requirements", [])
        elif kind == "conda":
            pip_requirements = self.environment.setdefault("pip", [])
        else:
            raise FalServerlessError(
                "Only {conda, virtualenv, container} "
                "are supported as environment options."
            )

        # Already has these.
        if set(pip_requirements).issuperset(set(requirements)):
            return None

        pip_requirements.extend(requirements)

    def get_exposed_port(self) -> int | None:
        if self.gateway.get("serve"):
            return _SERVE_PORT
        else:
            return self.gateway.get("exposed_port")


_DEFAULT_HOST = FalServerlessHost()
_SERVE_PORT = 8080

# Overload @function to help users identify the correct signature.
# NOTE: This is both in sync with host options and with environment configs from
# `isolate` package.


## virtualenv
### LocalHost
@overload
def function(
    kind: Literal["virtualenv"] = "virtualenv",
    *,
    python_version: str | None = None,
    requirements: list[str] | None = None,
    # Common options
    host: LocalHost,
    serve: Literal[False] = False,
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], IsolatedFunction[ArgsT, ReturnT]
]: ...


@overload
def function(
    kind: Literal["virtualenv"] = "virtualenv",
    *,
    python_version: str | None = None,
    requirements: list[str] | None = None,
    # Common options
    host: LocalHost,
    serve: Literal[True],
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], ServedIsolatedFunction[ArgsT, ReturnT]
]: ...


### FalServerlessHost
@overload
def function(
    kind: Literal["virtualenv"] = "virtualenv",
    *,
    python_version: str | None = None,
    requirements: list[str] | None = None,
    # Common options
    host: FalServerlessHost = _DEFAULT_HOST,
    serve: Literal[False] = False,
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
    # FalServerlessHost options
    metadata: dict[str, Any] | None = None,
    machine_type: str | list[str] = FAL_SERVERLESS_DEFAULT_MACHINE_TYPE,
    num_gpus: int | None = None,
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    max_multiplexing: int = FAL_SERVERLESS_DEFAULT_MAX_MULTIPLEXING,
    min_concurrency: int = FAL_SERVERLESS_DEFAULT_MIN_CONCURRENCY,
    request_timeout: int | None = None,
    setup_function: Callable[..., None] | None = None,
    _base_image: str | None = None,
    _scheduler: str | None = None,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], IsolatedFunction[ArgsT, ReturnT]
]: ...


@overload
def function(
    kind: Literal["virtualenv"] = "virtualenv",
    *,
    python_version: str | None = None,
    requirements: list[str] | None = None,
    # Common options
    host: FalServerlessHost = _DEFAULT_HOST,
    serve: Literal[True],
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
    # FalServerlessHost options
    metadata: dict[str, Any] | None = None,
    machine_type: str | list[str] = FAL_SERVERLESS_DEFAULT_MACHINE_TYPE,
    num_gpus: int | None = None,
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    max_multiplexing: int = FAL_SERVERLESS_DEFAULT_MAX_MULTIPLEXING,
    min_concurrency: int = FAL_SERVERLESS_DEFAULT_MIN_CONCURRENCY,
    request_timeout: int | None = None,
    setup_function: Callable[..., None] | None = None,
    _base_image: str | None = None,
    _scheduler: str | None = None,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], ServedIsolatedFunction[ArgsT, ReturnT]
]: ...


## conda
### LocalHost
@overload
def function(
    kind: Literal["conda"],
    *,
    python_version: str | None = None,
    env_dict: dict[str, Any] | None = None,
    env_yml: PathLike | str | None = None,
    env_yml_str: str | None = None,
    packages: list[str] | None = None,
    pip: list[str] | None = None,
    channels: list[str] | None = None,
    # Common options
    host: LocalHost,
    serve: Literal[False] = False,
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], IsolatedFunction[ArgsT, ReturnT]
]: ...


@overload
def function(
    kind: Literal["conda"],
    *,
    python_version: str | None = None,
    env_dict: dict[str, Any] | None = None,
    env_yml: PathLike | str | None = None,
    env_yml_str: str | None = None,
    packages: list[str] | None = None,
    pip: list[str] | None = None,
    channels: list[str] | None = None,
    # Common options
    host: LocalHost,
    serve: Literal[True],
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], ServedIsolatedFunction[ArgsT, ReturnT]
]: ...


### FalServerlessHost
@overload
def function(
    kind: Literal["conda"],
    *,
    python_version: str | None = None,
    env_dict: dict[str, Any] | None = None,
    env_yml: PathLike | str | None = None,
    env_yml_str: str | None = None,
    packages: list[str] | None = None,
    pip: list[str] | None = None,
    channels: list[str] | None = None,
    # Common options
    host: FalServerlessHost = _DEFAULT_HOST,
    serve: Literal[False] = False,
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
    # FalServerlessHost options
    metadata: dict[str, Any] | None = None,
    machine_type: str | list[str] = FAL_SERVERLESS_DEFAULT_MACHINE_TYPE,
    num_gpus: int | None = None,
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    max_multiplexing: int = FAL_SERVERLESS_DEFAULT_MAX_MULTIPLEXING,
    min_concurrency: int = FAL_SERVERLESS_DEFAULT_MIN_CONCURRENCY,
    request_timeout: int | None = None,
    setup_function: Callable[..., None] | None = None,
    _base_image: str | None = None,
    _scheduler: str | None = None,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], IsolatedFunction[ArgsT, ReturnT]
]: ...


@overload
def function(
    kind: Literal["conda"],
    *,
    python_version: str | None = None,
    env_dict: dict[str, Any] | None = None,
    env_yml: PathLike | str | None = None,
    env_yml_str: str | None = None,
    packages: list[str] | None = None,
    pip: list[str] | None = None,
    channels: list[str] | None = None,
    # Common options
    host: FalServerlessHost = _DEFAULT_HOST,
    serve: Literal[True],
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
    # FalServerlessHost options
    metadata: dict[str, Any] | None = None,
    machine_type: str | list[str] = FAL_SERVERLESS_DEFAULT_MACHINE_TYPE,
    num_gpus: int | None = None,
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    max_multiplexing: int = FAL_SERVERLESS_DEFAULT_MAX_MULTIPLEXING,
    min_concurrency: int = FAL_SERVERLESS_DEFAULT_MIN_CONCURRENCY,
    request_timeout: int | None = None,
    setup_function: Callable[..., None] | None = None,
    _base_image: str | None = None,
    _scheduler: str | None = None,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], ServedIsolatedFunction[ArgsT, ReturnT]
]: ...


@overload
def function(
    kind: Literal["container"],
    *,
    image: ContainerImage | None = None,
    # Common options
    host: FalServerlessHost = _DEFAULT_HOST,
    serve: Literal[False] = False,
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
    # FalServerlessHost options
    metadata: dict[str, Any] | None = None,
    machine_type: str | list[str] = FAL_SERVERLESS_DEFAULT_MACHINE_TYPE,
    num_gpus: int | None = None,
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    max_multiplexing: int = FAL_SERVERLESS_DEFAULT_MAX_MULTIPLEXING,
    min_concurrency: int = FAL_SERVERLESS_DEFAULT_MIN_CONCURRENCY,
    request_timeout: int | None = None,
    setup_function: Callable[..., None] | None = None,
    _base_image: str | None = None,
    _scheduler: str | None = None,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], IsolatedFunction[ArgsT, ReturnT]
]: ...


@overload
def function(
    kind: Literal["container"],
    *,
    image: ContainerImage | None = None,
    # Common options
    host: FalServerlessHost = _DEFAULT_HOST,
    serve: Literal[True],
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
    # FalServerlessHost options
    metadata: dict[str, Any] | None = None,
    machine_type: str | list[str] = FAL_SERVERLESS_DEFAULT_MACHINE_TYPE,
    num_gpus: int | None = None,
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    max_multiplexing: int = FAL_SERVERLESS_DEFAULT_MAX_MULTIPLEXING,
    min_concurrency: int = FAL_SERVERLESS_DEFAULT_MIN_CONCURRENCY,
    request_timeout: int | None = None,
    setup_function: Callable[..., None] | None = None,
    _base_image: str | None = None,
    _scheduler: str | None = None,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], ServedIsolatedFunction[ArgsT, ReturnT]
]: ...


# implementation
def function(  # type: ignore
    kind: str = "virtualenv",
    *,
    host: Host = _DEFAULT_HOST,
    **config: Any,
):
    options = host.parse_options(kind=kind, **config)

    def wrapper(func: Callable[ArgsT, ReturnT]):
        include_modules_from(func)
        proxy = IsolatedFunction(
            host=host,
            raw_func=func,  # type: ignore
            options=options,
        )
        return wraps(func)(proxy)  # type: ignore

    return wrapper


class FalFastAPI(FastAPI):
    """
    A subclass of FastAPI that adds some fal-specific functionality.
    """

    def openapi(self) -> dict[str, Any]:
        """
        Build the OpenAPI specification for the served function.
        Attach needed metadata for a better integration to fal.
        """
        spec = super().openapi()
        self._mark_order_openapi(spec)
        return spec

    def _mark_order_openapi(self, spec: dict[str, Any]):
        """
        Add x-fal-order-* keys to the OpenAPI specification to help the rendering of UI.

        NOTE: We rely on the fact that fastapi and Python dicts keep the order of
        properties.
        """

        def mark_order(obj: dict[str, Any], key: str):
            if key in obj:
                obj[f"x-fal-order-{key}"] = list(obj[key].keys())

        mark_order(spec, "paths")

        def order_schema_object(schema: dict[str, Any]):
            """
            Mark the order of properties in the schema object.
            They can have 'allOf', 'properties' or '$ref' key.
            """
            for sub_schema in schema.get("allOf", []):
                order_schema_object(sub_schema)

            mark_order(schema, "properties")

        for key in spec.get("components", {}).get("schemas", {}):
            order_schema_object(spec["components"]["schemas"][key])

        return spec


class RouteSignature(NamedTuple):
    path: str
    is_websocket: bool = False
    input_modal: type | None = None
    buffering: int | None = None
    session_timeout: float | None = None
    max_batch_size: int = 1
    emit_timings: bool = False


class BaseServable:
    version: ClassVar[str] = "unknown"

    def collect_routes(self) -> dict[RouteSignature, Callable[..., Any]]:
        raise NotImplementedError

    def _add_extra_middlewares(self, app: FastAPI):
        """
        For subclasses to add extra middlewares to the app.
        """
        pass

    def _add_extra_routes(self, app: FastAPI):
        pass

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        yield

    def _build_app(self) -> FastAPI:
        import json
        import traceback

        from fastapi import HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        from starlette_exporter import PrometheusMiddleware

        _app = FalFastAPI(
            lifespan=self.lifespan,
            root_path=os.getenv("FAL_APP_ROOT_PATH") or "",
        )

        _app.add_middleware(
            CORSMiddleware,
            allow_credentials=True,
            allow_headers=("*"),
            allow_methods=("*"),
            allow_origins=("*"),
        )
        _app.add_middleware(
            PrometheusMiddleware,
            prefix="http",
            group_paths=True,
            filter_unhandled_paths=True,
            app_name="fal",
        )

        self._add_extra_middlewares(_app)

        @_app.exception_handler(404)
        async def not_found_exception_handler(request: Request, exc: HTTPException):
            # Rewrite the message to include the path that was not found.
            # This is supposed to make it easier to understand to the user
            # that the error comes from the app and not our platform.
            if exc.detail == "Not Found":
                return JSONResponse(
                    {"detail": f"Path {request.url.path} not found"}, 404
                )
            else:
                # If it's not a generic 404, just return the original message.
                return JSONResponse({"detail": exc.detail}, 404)

        @_app.exception_handler(AppException)
        async def app_exception_handler(request: Request, exc: AppException):
            return JSONResponse({"detail": exc.message}, exc.status_code)

        @_app.exception_handler(FieldException)
        async def field_exception_handler(request: Request, exc: FieldException):
            return JSONResponse(exc.to_pydantic_format(), exc.status_code)

        @_app.exception_handler(CUDAOutOfMemoryException)
        async def cuda_out_of_memory_exception_handler(
            request: Request, exc: CUDAOutOfMemoryException
        ):
            return JSONResponse({"detail": exc.message}, exc.status_code)

        @_app.exception_handler(Exception)
        async def traceback_logging_exception_handler(request: Request, exc: Exception):
            _, MINOR, *_ = sys.version_info

            # traceback.format_exception() has a different signature in Python >=3.10
            if MINOR >= 10:
                formatted_exception = traceback.format_exception(exc)  # type: ignore
            else:
                formatted_exception = traceback.format_exception(
                    type(exc), exc, exc.__traceback__
                )

            print(json.dumps({"traceback": "".join(formatted_exception[::-1])}))

            if _is_cuda_oom_exception(exc):
                return await cuda_out_of_memory_exception_handler(
                    request, CUDAOutOfMemoryException()
                )

            return JSONResponse({"detail": "Internal Server Error"}, 500)

        routes = self.collect_routes()
        if not routes:
            raise ValueError("An application must have at least one route!")

        for signature, endpoint in routes.items():
            if signature.is_websocket:
                _app.add_api_websocket_route(
                    signature.path,
                    endpoint,
                    name=endpoint.__name__,
                )
            else:
                _app.add_api_route(
                    signature.path,
                    endpoint,
                    name=endpoint.__name__,
                    methods=["POST"],
                )

        self._add_extra_routes(_app)

        return _app

    def openapi(self) -> dict[str, Any]:
        """
        Build the OpenAPI specification for the served function.
        Attach needed metadata for a better integration to fal.
        """
        return self._build_app().openapi()

    def serve(self) -> None:
        import asyncio

        from prometheus_client import Gauge
        from starlette_exporter import handle_metrics
        from uvicorn import Config

        # NOTE: this uses the global prometheus registry
        app_info = Gauge("fal_app_info", "Fal application information", ["version"])
        app_info.labels(version=self.version).set(1)

        app = self._build_app()
        server = Server(
            config=Config(app, host="0.0.0.0", port=8080, timeout_keep_alive=300)
        )
        metrics_app = FastAPI()
        metrics_app.add_route("/metrics", handle_metrics)
        metrics_server = Server(config=Config(metrics_app, host="0.0.0.0", port=9090))

        async def _serve() -> None:
            tasks = {
                asyncio.create_task(server.serve()): server,
                asyncio.create_task(metrics_server.serve()): metrics_server,
            }

            _, pending = await asyncio.wait(
                tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not pending:
                return

            # try graceful shutdown
            for task in pending:
                tasks[task].should_exit = True
            _, pending = await asyncio.wait(pending, timeout=2)
            if not pending:
                return

            for task in pending:
                task.cancel()
            await asyncio.wait(pending)

        with suppress(asyncio.CancelledError):
            asyncio.set_event_loop(asyncio.new_event_loop())
            asyncio.run(_serve())


class ServeWrapper(BaseServable):
    _func: Callable

    def __init__(self, func: Callable):
        self._func = func

    def collect_routes(self) -> dict[RouteSignature, Callable[..., Any]]:
        return {
            RouteSignature("/"): self._func,
        }

    def __call__(self, *args, **kwargs) -> None:
        if len(args) != 0 or len(kwargs) != 0:
            print(
                f"[warning] {self._func.__name__} function is served with no arguments."
            )

        self.serve()


@dataclass
class IsolatedFunction(Generic[ArgsT, ReturnT]):
    host: Host[ArgsT, ReturnT]
    raw_func: Callable[ArgsT, ReturnT]
    options: Options
    executor: ThreadPoolExecutor = field(default_factory=ThreadPoolExecutor)
    reraise: bool = True

    def __getstate__(self) -> dict[str, Any]:
        # Ensure that the executor is not pickled.
        state = self.__dict__.copy()
        del state["executor"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "executor"):
            self.executor = ThreadPoolExecutor()

    def submit(self, *args: ArgsT.args, **kwargs: ArgsT.kwargs):
        # TODO: This should probably live inside each host since they can
        # have more optimized Future implementations (e.g. instead of real
        # threads, they can use state observers and detached runs).

        future = self.executor.submit(
            self.host.run,
            func=self.func,
            options=self.options,
            args=args,
            kwargs=kwargs,
        )
        return future

    def spawn(self, *args: ArgsT.args, **kwargs: ArgsT.kwargs):
        return self.host.spawn(
            self.func,
            self.options,
            args,
            kwargs,
        )

    def __call__(self, *args: ArgsT.args, **kwargs: ArgsT.kwargs) -> ReturnT:
        try:
            return self.host.run(
                self.func,
                self.options,
                args=args,
                kwargs=kwargs,
            )
        except FalMissingDependencyError as e:
            pairs = list(find_missing_dependencies(self.func, self.options.environment))
            if not pairs:
                raise e
            else:
                lines = []
                for used_modules, references in pairs:
                    lines.append(
                        f"\t- {used_modules!r} "
                        f"(accessed through {', '.join(map(repr, references))})"
                    )

                function_name = self.func.__name__
                raise FalServerlessError(
                    f"Couldn't deserialize your function on the remote server. \n\n"
                    f"[Hint] {function_name!r} function uses the following modules "
                    "which weren't present in the environment definition:\n"
                    + "\n".join(lines)
                ) from None
        except Exception as exc:
            cause = exc.__cause__
            if self.reraise and isinstance(exc, UserFunctionException) and cause:
                # re-raise original exception without our wrappers
                raise cause
            raise

    @overload
    def on(
        self, host: Host | None = None, *, serve: Literal[False] = False, **config: Any
    ) -> IsolatedFunction[ArgsT, ReturnT]: ...

    @overload
    def on(
        self, host: Host | None = None, *, serve: Literal[True], **config: Any
    ) -> ServedIsolatedFunction[ArgsT, ReturnT]: ...

    def on(self, host: Host | None = None, **config: Any):  # type: ignore
        host = host or self.host
        previous_host_options = {
            # Only keep the options that are supported by the new host.
            key: self.options.host.get(key)
            for key in host._SUPPORTED_KEYS
            if self.options.host.get(key) is not None
        }

        # The order of the options is important here (the latter
        # options override the former ones).
        host_options = {
            # All the previous options
            **previous_host_options,
            **self.options.environment,
            **self.options.gateway,
            # The new options
            **config,
        }
        new_options = host.parse_options(**host_options)
        return replace(
            self,
            host=host,
            options=new_options,
        )

    @property
    def func(self) -> Callable[ArgsT, ReturnT]:
        serve_mode = self.options.gateway.get("serve")
        if serve_mode:
            # This type can be safely ignored because this case only happens when it
            # is a ServedIsolatedFunction
            serve_func: Callable[[], None] = ServeWrapper(self.raw_func)
            return serve_func  # type: ignore
        else:
            return self.raw_func


if sys.version_info <= (3, 10):
    compatible_class = IsolatedFunction[Literal[None], None]  # type: ignore
else:
    compatible_class = IsolatedFunction[[], None]


class ServedIsolatedFunction(
    Generic[ArgsT, ReturnT],
    compatible_class,  # type: ignore
):
    # Class for type hinting purposes only.
    @overload  # type: ignore[override,no-overload-impl]
    def on(  # type: ignore[no-overload-impl]
        self, host: Host | None = None, *, serve: Literal[True] = True, **config: Any
    ) -> ServedIsolatedFunction[ArgsT, ReturnT]: ...

    @overload
    def on(
        self, host: Host | None = None, *, serve: Literal[False], **config: Any
    ) -> IsolatedFunction[ArgsT, ReturnT]: ...


class Server(uvicorn.Server):
    """Server is a uvicorn.Server that actually plays nicely with signals.
    By default, uvicorn's Server class overwrites the signal handler for SIGINT,
    swallowing the signal and preventing other tasks from cancelling.
    This class allows the task to be gracefully cancelled using asyncio's built-in task
    cancellation or with an event, like aiohttp.
    """

    def install_signal_handlers(self) -> None:
        pass
