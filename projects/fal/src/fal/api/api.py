from __future__ import annotations

import asyncio
import inspect
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from difflib import get_close_matches
from functools import wraps
from os import PathLike
from queue import Queue
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    Literal,
    NamedTuple,
    Optional,
    TypeVar,
    cast,
    overload,
)

import cloudpickle
import grpc
import tblib
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi import __version__ as fastapi_version
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from isolate.backends.common import Requirements
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from pydantic import __version__ as pydantic_version
from typing_extensions import Concatenate, ParamSpec

import fal.flags as flags
from fal._serialization import include_module, include_modules_from, patch_pickle
from fal.console import console
from fal.container import ContainerImage
from fal.exceptions import (
    AppException,
    CUDAOutOfMemoryException,
    FalServerlessException,
    FieldException,
)
from fal.exceptions._cuda import _is_cuda_oom_exception
from fal.file_sync import FileSync, FileSyncOptions
from fal.logging.isolate import IsolateLogPrinter
from fal.sdk import (
    FAL_SERVERLESS_DEFAULT_CONCURRENCY_BUFFER,
    FAL_SERVERLESS_DEFAULT_CONCURRENCY_BUFFER_PERC,
    FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    FAL_SERVERLESS_DEFAULT_MAX_MULTIPLEXING,
    FAL_SERVERLESS_DEFAULT_MIN_CONCURRENCY,
    AuthModeLiteral,
    Credentials,
    DeploymentStrategyLiteral,
    FalServerlessClient,
    FalServerlessConnection,
    File,
    HealthCheck,
    HostedRunState,
    MachineRequirements,
    RegisterApplicationResult,
    get_agent_credentials,
    get_default_credentials,
)

if TYPE_CHECKING:
    from isolate.backends import BaseEnvironment

ArgsT = ParamSpec("ArgsT")
ReturnT = TypeVar("ReturnT", covariant=True)  # noqa: PLC0105

BasicConfig = Dict[str, Any]


def merge_basic_config(target: BasicConfig, incoming: BasicConfig) -> None:
    for key, value in incoming.items():
        if key in target:
            continue
        target[key] = value


_UNSET = object()

SERVE_REQUIREMENTS = [
    f"fastapi=={fastapi_version}",
    f"pydantic=={pydantic_version}",
    f"tblib=={tblib.__version__}",
    "uvicorn",
    "starlette_exporter",
    # workaround for prometheus_client 0.23.0
    # https://github.com/prometheus/client_python/issues/1135
    "packaging",
    "structlog",
    "tomli",
    "tomli-w",
]


@dataclass
class FalServerlessError(FalServerlessException):
    message: str


@dataclass
class InternalFalServerlessError(FalServerlessException):
    message: str


@dataclass
class FalSerializationError(FalServerlessException):
    message: str


@dataclass
class FalMissingDependencyError(FalSerializationError): ...


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

    @property
    def application(self):
        from urllib.parse import urlparse  # noqa: PLC0415

        return urlparse(self.url).path.strip("/")


@dataclass
class Host(Generic[ArgsT, ReturnT]):
    """The physical environment where the isolated code
    is executed."""

    _SUPPORTED_KEYS: ClassVar[frozenset[str]] = frozenset()
    _GATEWAY_KEYS: ClassVar[frozenset[str]] = frozenset({"serve", "exposed_port"})
    _VIRTUALENV_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "python_version",
            "requirements",
            "resolver",
        }
    )
    _CONTAINER_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"image", "python_version", "requirements", "resolver", "force"}
    )
    _CONDA_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "python_version",
            "env_dict",
            "env_yml_str",
            "packages",
            "pip",
            "channels",
        }
    )
    _ENVIRONMENT_KEYS_BY_KIND: ClassVar[dict[str, frozenset[str]]] = {
        "virtualenv": _VIRTUALENV_KEYS,
        "container": _CONTAINER_KEYS,
        "conda": _CONDA_KEYS,
    }

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
        elif key == "image" and isinstance(value, ContainerImage):
            return "image", value.to_dict()
        else:
            return key, value

    @classmethod
    def parse_options(cls, **config: Any) -> Options:
        """Split the given set of options into host and
        environment options."""

        options = Options()
        kind = config.get("kind", "virtualenv")
        environment_keys = cls._ENVIRONMENT_KEYS_BY_KIND.get(kind)
        if environment_keys is None:
            supported_kinds = ", ".join(cls._ENVIRONMENT_KEYS_BY_KIND.keys())
            raise ValueError(
                f"Unrecognised environment kind {kind!r}. "
                f"Only {supported_kinds} are supported."
            )

        valid_environment_keys = {"kind", *environment_keys}
        all_supported_keys = {
            "kind",
            *cls._SUPPORTED_KEYS,
            *cls._GATEWAY_KEYS,
            *cls._VIRTUALENV_KEYS,
            *cls._CONTAINER_KEYS,
            *cls._CONDA_KEYS,
        }

        for item in config.items():
            key, value = cls.parse_key(*item)
            if key in cls._SUPPORTED_KEYS:
                options.host[key] = value
            elif key in cls._GATEWAY_KEYS:
                options.gateway[key] = value
            elif key in valid_environment_keys:
                options.environment[key] = value
            elif key in all_supported_keys:
                supported_keys = ", ".join(
                    f"{supported_key!r}" for supported_key in sorted(environment_keys)
                )
                raise ValueError(
                    f"Unsupported option {key!r} for environment kind {kind!r}. "
                    f"Supported keys for this kind are: {supported_keys}."
                )
            else:
                closest_match = get_close_matches(key, sorted(all_supported_keys), n=1)
                hint = f" Did you mean {closest_match[0]!r}?" if closest_match else ""
                raise ValueError(f"Unrecognised option {key!r}.{hint}")

        if options.gateway.get("serve"):
            options.add_requirements(SERVE_REQUIREMENTS)

        return options

    def run(
        self,
        func: Callable[ArgsT, ReturnT],
        options: Options,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        application_name: str | None = None,
        application_auth_mode: AuthModeLiteral | None = None,
    ) -> ReturnT:
        """Run the given function in the isolated environment."""
        raise NotImplementedError

    def spawn(
        self,
        func: Callable[ArgsT, ReturnT],
        options: Options,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        application_name: str | None = None,
        application_auth_mode: AuthModeLiteral | None = None,
    ) -> SpawnInfo:
        raise NotImplementedError


def cached(func: Callable[ArgsT, ReturnT]) -> Callable[ArgsT, ReturnT]:
    """Cache the result of the given function in-memory."""
    import hashlib  # noqa: PLC0415

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
        from functools import lru_cache  # noqa: PLC0415

        import isolate  # noqa: PLC0415

        # HACK: Using the isolate module as a global cache.

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


def _prepare_environment() -> BaseEnvironment:
    import isolate  # noqa: PLC0415

    return isolate.prepare_environment(
        "virtualenv",
        requirements=[
            f"cloudpickle=={cloudpickle.__version__}",
            f"tblib=={tblib.__version__}",
        ],
    )


@dataclass
class LocalHost(Host):
    # The environment which provides the default set of
    # packages for isolate agent to run.
    _AGENT_ENVIRONMENT: BaseEnvironment = field(default_factory=_prepare_environment)
    _log_printer = IsolateLogPrinter(debug=flags.DEBUG)

    def run(
        self,
        func: Callable[..., ReturnT],
        options: Options,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        application_name: str | None = None,
        application_auth_mode: AuthModeLiteral | None = None,
    ) -> ReturnT:
        import isolate  # noqa: PLC0415
        from isolate.backends.settings import DEFAULT_SETTINGS  # noqa: PLC0415
        from isolate.connections import PythonIPC  # noqa: PLC0415

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
            from isolate.connections.common import SerializationError  # noqa: PLC0415

            try:
                return fn(*args, **kwargs)
            except grpc.RpcError as e:
                msg = e.details() or str(e)
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    raise FalServerlessError(
                        "Could not reach fal host. "
                        "This is most likely a transient problem. "
                        "If it persists, please reach out to support@fal.ai with the following details: "  # noqa: E501
                        f"{msg}"
                    )
                elif msg.endswith("died with <Signals.SIGKILL: 9>.`."):
                    raise FalServerlessError(
                        "Isolated function crashed. "
                        "This is likely due to resource overflow. "
                        "You can try again by setting a bigger `machine_type`"
                    )
                elif e.code() == grpc.StatusCode.INVALID_ARGUMENT and (
                    "The function function could not be deserialized" in msg
                ):
                    raise FalMissingDependencyError(msg) from None
                else:
                    raise FalServerlessError(msg)
            except SerializationError as e:
                msg = str(e)
                cause = e.__cause__
                if isinstance(cause, ModuleNotFoundError):
                    missing_module = cause.name
                    msg += (
                        f". Could not find module '{missing_module}'. "
                        "This is likely due to a missing dependency. "
                        "Please make sure to include all dependencies "
                        "in the environment configuration."
                    )
                raise FalSerializationError(msg) from cause

        return handler

    return decorator


def find_missing_dependencies(
    func: Callable, env: dict
) -> Iterator[tuple[str, list[str]]]:
    import dill  # noqa: PLC0415

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
    requirements = Requirements.from_raw(raw_requirements)
    specified_requirements = set()
    for layer in requirements.layers:
        for raw_requirement in layer:
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
            "regions",
            "num_gpus",
            "keep_alive",
            "max_concurrency",
            "min_concurrency",
            "concurrency_buffer",
            "concurrency_buffer_perc",
            "scaling_delay",
            "max_multiplexing",
            "setup_function",
            "metadata",
            "request_timeout",
            "startup_timeout",
            "private_logs",
            "_base_image",
            "_scheduler",
            "_scheduler_options",
            "app_files",
            "app_files_ignore",
            "app_files_context_dir",
            "health_check_config",
            "skip_retry_conditions",
            "termination_grace_period_seconds",
        }
    )

    url: str = FAL_SERVERLESS_DEFAULT_URL
    local_file_path: str = ""
    credentials: Credentials = field(default_factory=get_default_credentials)
    environment_name: Optional[str] = None

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    _log_printer: IsolateLogPrinter = field(
        default_factory=lambda: IsolateLogPrinter(debug=flags.DEBUG), init=False
    )

    _thread_pool: ThreadPoolExecutor = field(
        default_factory=ThreadPoolExecutor, init=False
    )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_thread_pool"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._thread_pool = ThreadPoolExecutor()
        self.credentials = get_agent_credentials(self.credentials)

    @property
    def _connection(self) -> FalServerlessConnection:
        with self._lock:
            client = FalServerlessClient(self.url, self.credentials)
            return client.connect()

    def files_sync(self, options: FileSyncOptions) -> list[File]:
        """Sync files to the server."""
        # Auto-exclude the app file, it gets serialized separately
        if self.local_file_path and options.files_list:
            import re  # noqa: PLC0415
            from pathlib import Path  # noqa: PLC0415

            context = Path(options.files_context_dir or ".").resolve()
            app_file = Path(self.local_file_path).resolve()
            if app_file.is_relative_to(context):
                rel_path = str(app_file.relative_to(context))
                options.files_ignore.append(re.compile(f"^{re.escape(rel_path)}$"))

        res = []
        if options.files_list:
            sync = FileSync(self.local_file_path)
            files, errors = sync.sync_files(
                options.files_list,
                files_ignore=options.files_ignore,
                files_context_dir=options.files_context_dir,
            )
            if errors:
                for error in errors:
                    console.print(
                        f"Error uploading file {error.relative_path}: {error.message}"
                    )

                raise FalServerlessException("Error uploading files")

            res = [
                File(relative_path=file.relative_path, hash=file.hash) for file in files
            ]

        return res

    @_handle_grpc_error()
    def register(
        self,
        func: Callable[ArgsT, ReturnT],
        options: Options,
        *,
        application_name: Optional[str] = None,
        application_auth_mode: Optional[AuthModeLiteral] = None,
        source_code: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        deployment_strategy: DeploymentStrategyLiteral,
        scale: bool = True,
        environment_name: Optional[str] = None,
    ) -> Optional[RegisterApplicationResult]:
        from isolate.backends.common import active_python  # noqa: PLC0415

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
        concurrency_buffer = options.host.get("concurrency_buffer")
        concurrency_buffer_perc = options.host.get("concurrency_buffer_perc")
        scaling_delay = options.host.get("scaling_delay")
        max_multiplexing = options.host.get("max_multiplexing")
        exposed_port = options.get_exposed_port()
        request_timeout = options.host.get("request_timeout")
        startup_timeout = options.host.get("startup_timeout")
        regions = options.host.get("regions")
        health_check_config = options.host.get("health_check_config")
        skip_retry_conditions = options.host.get("skip_retry_conditions")
        termination_grace_period_seconds = options.host.get(
            "termination_grace_period_seconds"
        )
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
            concurrency_buffer=concurrency_buffer,
            concurrency_buffer_perc=concurrency_buffer_perc,
            scaling_delay=scaling_delay,
            request_timeout=request_timeout,
            startup_timeout=startup_timeout,
            valid_regions=regions,
        )

        health_check_config = options.host.get("health_check_config")

        files = self.files_sync(FileSyncOptions.from_options(options))

        partial_func = _prepare_partial_func(func)

        if metadata is None:
            metadata = {}

        # TODO: let the user send more metadata than just openapi
        if isinstance(func, ServeWrapper):
            # Assigning in a separate property leaving a place for the user
            # to add more metadata in the future
            metadata["openapi"] = func.openapi()

        for partial_result in self._connection.register(
            partial_func,
            environments,
            application_name=application_name,
            auth_mode=application_auth_mode,
            source_code=source_code,
            machine_requirements=machine_requirements,
            metadata=metadata,
            deployment_strategy=deployment_strategy,
            scale=scale,
            health_check_config=health_check_config,
            # By default, logs are public
            private_logs=options.host.get("private_logs", False),
            files=files,
            skip_retry_conditions=skip_retry_conditions,
            environment_name=environment_name,
            termination_grace_period_seconds=termination_grace_period_seconds,
        ):
            for log in partial_result.logs:
                self._log_printer.print(log)

            if partial_result.result:
                return partial_result

        return None

    @_handle_grpc_error()
    def _run(
        self,
        func: Callable[..., ReturnT],
        options: Options,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result_handler: Callable[..., None],
        application_name: str | None = None,
        application_auth_mode: AuthModeLiteral | None = None,
    ) -> ReturnT:
        from isolate.backends.common import active_python  # noqa: PLC0415

        environment_options = options.environment.copy()
        environment_options.setdefault("python_version", active_python())
        environments = [self._connection.define_environment(**environment_options)]

        machine_type: list[str] | str = options.host.get(
            "machine_type", FAL_SERVERLESS_DEFAULT_MACHINE_TYPE
        )
        keep_alive = options.host.get("keep_alive", FAL_SERVERLESS_DEFAULT_KEEP_ALIVE)
        max_concurrency = options.host.get("max_concurrency")
        min_concurrency = options.host.get("min_concurrency")
        concurrency_buffer = options.host.get("concurrency_buffer")
        concurrency_buffer_perc = options.host.get("concurrency_buffer_perc")
        scaling_delay = options.host.get("scaling_delay")
        max_multiplexing = options.host.get("max_multiplexing")
        base_image = options.host.get("_base_image", None)
        scheduler = options.host.get("_scheduler", None)
        scheduler_options = options.host.get("_scheduler_options", None)
        exposed_port = options.get_exposed_port()
        setup_function = options.host.get("setup_function", None)
        request_timeout = options.host.get("request_timeout")
        startup_timeout = options.host.get("startup_timeout")
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
            concurrency_buffer=concurrency_buffer,
            concurrency_buffer_perc=concurrency_buffer_perc,
            scaling_delay=scaling_delay,
            request_timeout=request_timeout,
            startup_timeout=startup_timeout,
        )

        files = self.files_sync(FileSyncOptions.from_options(options))

        return_value = _UNSET
        # Allow isolate provided arguments (such as setup function) to take
        # precedence over the ones provided by the user.
        partial_func = _prepare_partial_func(func, *args, **kwargs)
        effective_app_name = application_name or getattr(func, "__name__", None)
        effective_auth_mode = application_auth_mode or "public"

        for partial_result in self._connection.run(
            partial_func,
            environments,
            machine_requirements=machine_requirements,
            setup_function=setup_function,
            files=files,
            application_name=effective_app_name,
            auth_mode=effective_auth_mode,
            environment_name=self.environment_name,
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
        application_name: str | None = None,
        application_auth_mode: AuthModeLiteral | None = None,
    ) -> ReturnT:
        effective_auth_mode = application_auth_mode or "public"

        def result_handler(partial_result):
            if service_urls := partial_result.service_urls:
                from rich.rule import Rule  # noqa: PLC0415
                from rich.text import Text  # noqa: PLC0415

                from fal.flags import URL_OUTPUT  # noqa: PLC0415

                print("")

                # Build panel content with grouped sections
                lines = Text()
                endpoints = getattr(func, "_routes", ["/"])  # type: ignore[attr-defined]

                AUTH_EXPLANATIONS = {
                    "public": "no authentication required",
                    "private": "only you/team can access",
                    "shared": "any authenticated user can access",
                }
                auth_desc = AUTH_EXPLANATIONS.get(
                    effective_auth_mode, effective_auth_mode
                )
                lines.append(f"▸ Auth: {effective_auth_mode} ", style="bold")
                lines.append(f"({auth_desc})\n\n", style="dim")

                # Playground section
                if URL_OUTPUT != "none":
                    lines.append("▸ Playground ", style="bold")
                    lines.append("(open in browser)\n", style="dim")
                    for endpoint in endpoints:
                        lines.append(
                            f"  {service_urls.playground}{endpoint}\n", style="cyan"
                        )

                # API Endpoints section
                if URL_OUTPUT == "all":
                    lines.append("\n")
                    lines.append("▸ API Endpoints ", style="bold")
                    lines.append("(use in code)\n", style="dim")
                    for endpoint in endpoints:
                        lines.append(
                            f"  Sync   {service_urls.run}{endpoint}\n", style="cyan"
                        )
                        lines.append(
                            f"  Async  {service_urls.queue}{endpoint}\n", style="cyan"
                        )

                title = Text(f"Ephemeral App ({effective_auth_mode})", style="bold")
                subtitle = Text("Deleted when process exits", style="dim")
                console.print(Rule(title, style="green"))
                console.print(lines)
                console.print(Rule(subtitle, style="green"))

            for log in partial_result.logs:
                if (
                    "Access the playground at" in log.message
                    or "And API access through" in log.message
                ):
                    # Obsolete messages from before service_urls were added.
                    continue
                self._log_printer.print(log)

        return self._run(
            func,
            options,
            args,
            kwargs,
            result_handler=result_handler,
            application_name=application_name,
            application_auth_mode=application_auth_mode,
        )

    def spawn(
        self,
        func: Callable[..., ReturnT],
        options: Options,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        application_name: str | None = None,
        application_auth_mode: AuthModeLiteral | None = None,
    ) -> SpawnInfo:
        ret = SpawnInfo()

        def result_handler(partial_result):
            ret.stream = partial_result.stream
            if service_urls := partial_result.service_urls:
                ret.url = service_urls.run
            for log in partial_result.logs:
                if ret._url is None and "And API access through" in log.message:
                    ret.url = log.message.rsplit()[-1].replace("queue.", "")
                ret.logs.put(log)

        self._thread_pool.submit(
            self._run,
            func,
            options,
            args,
            kwargs,
            result_handler=result_handler,
            application_name=application_name,
            application_auth_mode=application_auth_mode,
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

        parsed = Requirements.from_raw(pip_requirements)
        existing = {req for layer in parsed.layers for req in layer}

        # Already has these.
        if existing.issuperset(set(requirements)):
            return None

        layered = (
            pip_requirements
            and isinstance(pip_requirements, list)
            and all(isinstance(item, list) for item in pip_requirements)
        )
        if layered:
            pip_requirements.append([*requirements])
        else:
            pip_requirements.extend(requirements)

    def get_exposed_port(self) -> int | None:
        if self.gateway.get("serve"):
            return _SERVE_PORT
        else:
            return self.gateway.get("exposed_port")


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
    requirements: list[str] | list[list[str]] | None = None,
    # Common options
    host: LocalHost,
    serve: Literal[False] = False,
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
    local_python_modules: list[str] | None = None,
    force_env_build: bool = False,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], IsolatedFunction[ArgsT, ReturnT]
]: ...


@overload
def function(
    kind: Literal["virtualenv"] = "virtualenv",
    *,
    python_version: str | None = None,
    requirements: list[str] | list[list[str]] | None = None,
    # Common options
    host: LocalHost,
    serve: Literal[True],
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
    local_python_modules: list[str] | None = None,
    force_env_build: bool = False,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], ServedIsolatedFunction[ArgsT, ReturnT]
]: ...


### FalServerlessHost
@overload
def function(
    kind: Literal["virtualenv"] = "virtualenv",
    *,
    python_version: str | None = None,
    requirements: list[str] | list[list[str]] | None = None,
    # Common options
    host: FalServerlessHost | None = None,
    serve: Literal[False] = False,
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
    local_python_modules: list[str] | None = None,
    # FalServerlessHost options
    metadata: dict[str, Any] | None = None,
    machine_type: str | list[str] = FAL_SERVERLESS_DEFAULT_MACHINE_TYPE,
    regions: list[str] | None = None,
    num_gpus: int | None = None,
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    max_multiplexing: int = FAL_SERVERLESS_DEFAULT_MAX_MULTIPLEXING,
    min_concurrency: int = FAL_SERVERLESS_DEFAULT_MIN_CONCURRENCY,
    concurrency_buffer: int = FAL_SERVERLESS_DEFAULT_CONCURRENCY_BUFFER,
    concurrency_buffer_perc: int = FAL_SERVERLESS_DEFAULT_CONCURRENCY_BUFFER_PERC,
    scaling_delay: int | None = None,
    request_timeout: int | None = None,
    startup_timeout: int | None = None,
    setup_function: Callable[..., None] | None = None,
    force_env_build: bool = False,
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
    requirements: list[str] | list[list[str]] | None = None,
    # Common options
    host: FalServerlessHost | None = None,
    serve: Literal[True],
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
    local_python_modules: list[str] | None = None,
    # FalServerlessHost options
    metadata: dict[str, Any] | None = None,
    machine_type: str | list[str] = FAL_SERVERLESS_DEFAULT_MACHINE_TYPE,
    regions: list[str] | None = None,
    num_gpus: int | None = None,
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    max_multiplexing: int = FAL_SERVERLESS_DEFAULT_MAX_MULTIPLEXING,
    min_concurrency: int = FAL_SERVERLESS_DEFAULT_MIN_CONCURRENCY,
    concurrency_buffer: int = FAL_SERVERLESS_DEFAULT_CONCURRENCY_BUFFER,
    concurrency_buffer_perc: int = FAL_SERVERLESS_DEFAULT_CONCURRENCY_BUFFER_PERC,
    scaling_delay: int | None = None,
    request_timeout: int | None = None,
    startup_timeout: int | None = None,
    setup_function: Callable[..., None] | None = None,
    force_env_build: bool = False,
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
    local_python_modules: list[str] | None = None,
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
    local_python_modules: list[str] | None = None,
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
    host: FalServerlessHost | None = None,
    serve: Literal[False] = False,
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
    local_python_modules: list[str] | None = None,
    # FalServerlessHost options
    metadata: dict[str, Any] | None = None,
    machine_type: str | list[str] = FAL_SERVERLESS_DEFAULT_MACHINE_TYPE,
    regions: list[str] | None = None,
    num_gpus: int | None = None,
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    max_multiplexing: int = FAL_SERVERLESS_DEFAULT_MAX_MULTIPLEXING,
    min_concurrency: int = FAL_SERVERLESS_DEFAULT_MIN_CONCURRENCY,
    concurrency_buffer: int = FAL_SERVERLESS_DEFAULT_CONCURRENCY_BUFFER,
    concurrency_buffer_perc: int = FAL_SERVERLESS_DEFAULT_CONCURRENCY_BUFFER_PERC,
    scaling_delay: int | None = None,
    request_timeout: int | None = None,
    startup_timeout: int | None = None,
    setup_function: Callable[..., None] | None = None,
    force_env_build: bool = False,
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
    host: FalServerlessHost | None = None,
    serve: Literal[True],
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
    local_python_modules: list[str] | None = None,
    # FalServerlessHost options
    metadata: dict[str, Any] | None = None,
    machine_type: str | list[str] = FAL_SERVERLESS_DEFAULT_MACHINE_TYPE,
    regions: list[str] | None = None,
    num_gpus: int | None = None,
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    max_multiplexing: int = FAL_SERVERLESS_DEFAULT_MAX_MULTIPLEXING,
    min_concurrency: int = FAL_SERVERLESS_DEFAULT_MIN_CONCURRENCY,
    concurrency_buffer: int = FAL_SERVERLESS_DEFAULT_CONCURRENCY_BUFFER,
    concurrency_buffer_perc: int = FAL_SERVERLESS_DEFAULT_CONCURRENCY_BUFFER_PERC,
    scaling_delay: int | None = None,
    request_timeout: int | None = None,
    startup_timeout: int | None = None,
    setup_function: Callable[..., None] | None = None,
    force_env_build: bool = False,
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
    host: FalServerlessHost | None = None,
    serve: Literal[False] = False,
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
    local_python_modules: list[str] | None = None,
    # FalServerlessHost options
    metadata: dict[str, Any] | None = None,
    machine_type: str | list[str] = FAL_SERVERLESS_DEFAULT_MACHINE_TYPE,
    regions: list[str] | None = None,
    num_gpus: int | None = None,
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    max_multiplexing: int = FAL_SERVERLESS_DEFAULT_MAX_MULTIPLEXING,
    min_concurrency: int = FAL_SERVERLESS_DEFAULT_MIN_CONCURRENCY,
    concurrency_buffer: int = FAL_SERVERLESS_DEFAULT_CONCURRENCY_BUFFER,
    concurrency_buffer_perc: int = FAL_SERVERLESS_DEFAULT_CONCURRENCY_BUFFER_PERC,
    scaling_delay: int | None = None,
    request_timeout: int | None = None,
    startup_timeout: int | None = None,
    setup_function: Callable[..., None] | None = None,
    force_env_build: bool = False,
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
    host: FalServerlessHost | None = None,
    serve: Literal[True],
    exposed_port: int | None = None,
    max_concurrency: int | None = None,
    local_python_modules: list[str] | None = None,
    # FalServerlessHost options
    metadata: dict[str, Any] | None = None,
    machine_type: str | list[str] = FAL_SERVERLESS_DEFAULT_MACHINE_TYPE,
    regions: list[str] | None = None,
    num_gpus: int | None = None,
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    max_multiplexing: int = FAL_SERVERLESS_DEFAULT_MAX_MULTIPLEXING,
    min_concurrency: int = FAL_SERVERLESS_DEFAULT_MIN_CONCURRENCY,
    concurrency_buffer: int = FAL_SERVERLESS_DEFAULT_CONCURRENCY_BUFFER,
    concurrency_buffer_perc: int = FAL_SERVERLESS_DEFAULT_CONCURRENCY_BUFFER_PERC,
    scaling_delay: int | None = None,
    request_timeout: int | None = None,
    startup_timeout: int | None = None,
    setup_function: Callable[..., None] | None = None,
    force_env_build: bool = False,
    _base_image: str | None = None,
    _scheduler: str | None = None,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], ServedIsolatedFunction[ArgsT, ReturnT]
]: ...


# implementation
def function(  # type: ignore
    kind: str = "virtualenv",
    *,
    host: Host | None = None,
    local_python_modules: list[str] | None = None,
    **config: Any,
):
    if host is None:
        host = FalServerlessHost()

    if "requirements" in config and config["requirements"] is not None:
        requirements = config["requirements"]
        is_str_list = isinstance(requirements, list) and all(
            isinstance(item, str) for item in requirements
        )
        is_str_list_list = isinstance(requirements, list) and all(
            isinstance(item, list) and all(isinstance(req, str) for req in item)
            for item in requirements
        )
        if not is_str_list and not is_str_list_list:
            raise ValueError(
                "requirements must be a list of strings or a list of lists of strings."
            )

    # NOTE: assuming kind="container" if image is provided
    if config.get("image"):
        kind = "container"

    if kind == "container" and config.get("app_files"):
        raise ValueError("app_files is not supported for container apps.")

    if config.get("force_env_build") is not None:
        force_env_build = config.pop("force_env_build")
        if kind == "container":
            config["force"] = force_env_build
        elif force_env_build:
            console.print(
                "[bold yellow]Note:[/bold yellow] [dim]--no-cache[/dim]"
                " is only supported for container apps as of now. Ignoring."
            )

    options = host.parse_options(kind=kind, **config)

    def wrapper(func: Callable[ArgsT, ReturnT]):
        include_modules_from(func)

        if local_python_modules and not isinstance(local_python_modules, list):
            raise ValueError(
                "local_python_modules must be a list of module names as strings, got "
                f"{repr(local_python_modules)}"
            )

        for idx, module_name in enumerate(local_python_modules or []):
            if not isinstance(module_name, str):
                raise ValueError(
                    "local_python_modules must be a list of module names as strings, "
                    f"got {repr(module_name)} at index {idx}"
                )
            include_module(module_name)

        proxy = IsolatedFunction(
            host=host,  # type: ignore
            raw_func=func,  # type: ignore
            options=options,
            app_name=getattr(func, "__name__", None),
        )
        return wraps(func)(proxy)  # type: ignore

    return wrapper


class FalFastAPI(FastAPI):
    """
    A subclass of FastAPI that adds some fal-specific functionality.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store websocket routes for OpenAPI injection
        self._websocket_routes: list[tuple[RouteSignature, Callable[..., Any]]] = []

    def add_websocket_route_with_metadata(
        self,
        signature: RouteSignature,
        endpoint: Callable[..., Any],
    ):
        """Add websocket route and store metadata for OpenAPI injection."""
        self.add_api_websocket_route(
            signature.path,
            endpoint,
            name=endpoint.__name__,
        )
        self._websocket_routes.append((signature, endpoint))

    def openapi(self) -> dict[str, Any]:
        """
        Build the OpenAPI specification for the served function.
        Attach needed metadata for a better integration to fal.
        """
        spec = super().openapi()
        self._mark_order_openapi(spec)
        self._inject_websocket_endpoints(spec)
        return spec

    def _ensure_components_schemas(self, spec: dict[str, Any]) -> dict[str, Any]:
        if "components" not in spec:
            spec["components"] = {}
        if "schemas" not in spec["components"]:
            spec["components"]["schemas"] = {}
        return spec["components"]["schemas"]

    def _ensure_path_item(self, spec: dict[str, Any], path: str) -> dict[str, Any]:
        if "paths" not in spec:
            spec["paths"] = {}
        if path not in spec["paths"]:
            spec["paths"][path] = {}
        return spec["paths"][path]

    def _register_model_schema(
        self, schemas: dict[str, Any], model: type | None
    ) -> str | None:
        if model is None:
            return None

        schema_name = model.__name__
        if pydantic_version.startswith("2."):
            from pydantic import TypeAdapter  # type: ignore  # noqa: PLC0415

            adapter = TypeAdapter(model)
            schema = adapter.json_schema(ref_template="#/components/schemas/{model}")
        else:
            from pydantic.schema import schema as pydantic_schema  # noqa: PLC0415

            schema = pydantic_schema([model], ref_prefix="#/components/schemas/")
            schema = schema.get("definitions", {}).get(schema_name, schema)

        if "$defs" in schema:
            for def_name, def_schema in schema["$defs"].items():
                schemas[def_name] = def_schema
            del schema["$defs"]

        schemas[schema_name] = schema
        return schema_name

    def _build_protocol_operation(
        self,
        signature: RouteSignature,
        display_endpoint: Callable[..., Any],
        input_schema_name: str | None,
        output_schema_name: str | None,
    ) -> dict[str, Any]:
        operation: dict[str, Any] = {
            "type": "realtime" if signature.realtime_mode else "websocket",
            "operationId": display_endpoint.__name__,
            "config": {
                "buffering": signature.buffering,
                "sessionTimeout": signature.session_timeout,
                "maxBatchSize": signature.max_batch_size,
            },
        }
        if signature.content_type is not None:
            operation["contentType"] = signature.content_type
        if signature.realtime_mode is not None:
            operation["realtimeMode"] = signature.realtime_mode
        return operation

    def _build_protocol_mirror_post(
        self,
        signature: RouteSignature,
        display_endpoint: Callable[..., Any],
        input_schema_name: str | None,
        output_schema_name: str | None,
    ) -> dict[str, Any]:
        endpoint_type = "Realtime" if signature.realtime_mode else "WebSocket"
        operation: dict[str, Any] = {
            "operationId": f"{display_endpoint.__name__}_post",
            "summary": f"{endpoint_type} endpoint: {display_endpoint.__name__}",
            "description": display_endpoint.__doc__ or "",
        }
        if input_schema_name:
            operation["requestBody"] = {
                "content": {
                    "application/json": {
                        "schema": {"$ref": f"#/components/schemas/{input_schema_name}"}
                    }
                }
            }
        if output_schema_name:
            operation["responses"] = {
                "200": {
                    "description": "WebSocket message response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": f"#/components/schemas/{output_schema_name}"
                            }
                        }
                    },
                }
            }
        else:
            operation["responses"] = {
                "204": {"description": "WebSocket message response"}
            }
        return operation

    def _inject_websocket_endpoints(self, spec: dict[str, Any]):
        """Inject WebSocket endpoint metadata using x-fal-protocol extension."""
        for signature, endpoint in self._websocket_routes:
            path = signature.path

            schemas = self._ensure_components_schemas(spec)
            input_schema_name = self._register_model_schema(
                schemas, signature.input_modal
            )
            output_schema_name = self._register_model_schema(
                schemas, signature.output_modal
            )
            path_item = self._ensure_path_item(spec, path)
            display_endpoint = getattr(endpoint, "original_func", endpoint)
            path_item["x-fal-protocol"] = self._build_protocol_operation(
                signature,
                display_endpoint,
                input_schema_name,
                output_schema_name,
            )

            if "post" not in path_item and (input_schema_name or output_schema_name):
                path_item["post"] = self._build_protocol_mirror_post(
                    signature,
                    display_endpoint,
                    input_schema_name,
                    output_schema_name,
                )

        # Update x-fal-order-paths to include websocket paths
        if self._websocket_routes:
            ws_paths = [sig.path for sig, _ in self._websocket_routes]
            existing_order = list(spec.get("x-fal-order-paths", []))
            for path in ws_paths:
                if path not in existing_order:
                    existing_order.append(path)
            spec["x-fal-order-paths"] = existing_order

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
    health_check: HealthCheck | None = None
    input_modal: type | None = None
    output_modal: type | None = None
    realtime_mode: str | None = None
    content_type: str | None = None
    buffering: int | None = None
    session_timeout: float | None = None
    max_batch_size: int = 1
    emit_timings: bool = False
    encode_message: Callable[[Any], bytes] | None = None
    decode_message: Callable[[bytes], Any] | None = None


class FalServer(uvicorn.Server):
    def set_handle_exit(self, handle_exit):
        self._handle_exit = handle_exit

    def handle_exit(self, sig, frame):
        super().handle_exit(sig, frame)
        try:
            self._handle_exit()
        except BaseException as e:
            from fastapi.logger import logger  # noqa: PLC0415

            logger.exception(f"Error in handle_exit: {e}")


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

    def _build_app(self) -> FalFastAPI:
        import json  # noqa: PLC0415
        import traceback  # noqa: PLC0415

        from fastapi import HTTPException, Request  # noqa: PLC0415
        from fastapi.middleware.cors import CORSMiddleware  # noqa: PLC0415
        from fastapi.responses import JSONResponse  # noqa: PLC0415
        from starlette_exporter import PrometheusMiddleware  # noqa: PLC0415

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
                # For 404 errors (non-existent endpoints), set billable units to 0.
                # This prevents users from being charged when they hit endpoints that
                # don't exist. Without this, the platform would use the default billable
                # units for the endpoint, incorrectly charging users for failed requests
                headers = dict(exc.headers) if exc.headers else {}
                headers["x-fal-billable-units"] = "0"
                return JSONResponse(
                    {"detail": f"Path {request.url.path} not found"},
                    404,
                    headers=headers,
                )
            else:
                # If it's not a generic 404, just return the original message.
                return JSONResponse({"detail": exc.detail}, 404)

        @_app.exception_handler(AppException)
        async def app_exception_handler(request: Request, exc: AppException):
            return JSONResponse({"detail": exc.message}, exc.status_code)

        @_app.exception_handler(FieldException)
        async def field_exception_handler(request: Request, exc: FieldException):
            headers = {}
            if exc.billable_units:
                # poor man's validation. we dont want people to pass in
                # non-numeric values.
                units_float = float(exc.billable_units)
                # we dont want to add 8 decimal places for ints.
                format_string = ".0f" if isinstance(exc.billable_units, int) else ".8f"
                headers["x-fal-billable-units"] = format(units_float, format_string)
            return JSONResponse(
                exc.to_pydantic_format(), exc.status_code, headers=headers
            )

        # ref: https://github.com/fastapi/fastapi/blob/37c8e7d76b4b47eb2c4cced6b4de59eb3d5f08eb/fastapi/exception_handlers.py#L20
        @_app.exception_handler(RequestValidationError)
        async def request_val_exception_handler(
            request: Request, exc: RequestValidationError
        ):
            return JSONResponse(
                {"detail": jsonable_encoder(exc.errors())},
                422,
                headers={"x-fal-billable-units": "0"},
            )

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

            print(
                json.dumps({"traceback": "".join(formatted_exception[::-1])}),
                flush=True,
            )

            if _is_cuda_oom_exception(exc):
                return await cuda_out_of_memory_exception_handler(
                    request, CUDAOutOfMemoryException()
                )

            # last line of defense against misc GPU errors that could indicate a bad
            # worker
            if any(marker in str(exc).lower() for marker in ["cuda", "cudnn", "nvml"]):
                return JSONResponse({"detail": "GPU error"}, 503)

            return JSONResponse({"detail": "Internal Server Error"}, 500)

        routes = self.collect_routes()
        if not routes:
            raise ValueError("An application must have at least one route!")

        for signature, endpoint in routes.items():
            if signature.is_websocket:
                _app.add_websocket_route_with_metadata(signature, endpoint)
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
        try:
            return self._build_app().openapi()
        except Exception as e:
            raise FalServerlessException(
                "Failed to generate OpenAPI metadata for function"
            ) from e

    def handle_exit(self):
        pass

    async def serve(self) -> None:
        from prometheus_client import Gauge  # noqa: PLC0415
        from starlette_exporter import handle_metrics  # noqa: PLC0415

        # NOTE: this uses the global prometheus registry
        app_info = Gauge("fal_app_info", "Fal application information", ["version"])
        app_info.labels(version=self.version).set(1)

        app = self._build_app()

        # We use the default workers=1 config because setup function can be heavy
        # and it runs once per worker.
        server = FalServer(
            config=uvicorn.Config(
                app, host="0.0.0.0", port=8080, timeout_keep_alive=300, lifespan="on"
            )
        )
        server.set_handle_exit(self.handle_exit)

        metrics_app = FastAPI()
        metrics_app.add_route("/metrics", handle_metrics)
        metrics_server = uvicorn.Server(
            config=uvicorn.Config(metrics_app, host="0.0.0.0", port=9090)
        )

        async def _serve() -> None:
            app_task = asyncio.create_task(server.serve())
            metrics_task = asyncio.create_task(metrics_server.serve())
            tasks = {app_task, metrics_task}

            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            from fastapi.logger import logger  # noqa: PLC0415

            if app_task in done and metrics_task in pending:
                metrics_task.cancel()

            app_exc, metrics_exc = await asyncio.gather(
                app_task,
                metrics_task,
                return_exceptions=True,
            )

            if app_exc:
                logger.error("App server exited with error", exc_info=app_exc)

            if metrics_exc and not isinstance(metrics_exc, asyncio.CancelledError):
                logger.error("Metrics server exited with error", exc_info=metrics_exc)

            if app_exc:
                raise app_exc

            # graceful termination and timeout should be handled by external scheduler.

        await _serve()


class ServeWrapper(BaseServable):
    _func: Callable

    def __init__(self, func: Callable):
        self._func = func

    def collect_routes(self) -> dict[RouteSignature, Callable[..., Any]]:
        return {
            RouteSignature("/"): self._func,
        }

    async def __call__(self, *args, **kwargs) -> None:
        if len(args) != 0 or len(kwargs) != 0:
            print(
                f"[warning] {self._func.__name__} function is served with no arguments."
            )

        await self.serve()


@dataclass
class IsolatedFunction(Generic[ArgsT, ReturnT]):
    host: Host[ArgsT, ReturnT]
    raw_func: Callable[ArgsT, ReturnT]
    options: Options
    executor: ThreadPoolExecutor = field(default_factory=ThreadPoolExecutor)
    reraise: bool = True
    app_name: str | None = None
    app_auth: AuthModeLiteral | None = None

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
            application_name=self.app_name,
            application_auth_mode=self.app_auth,
        )
        return future

    def spawn(self, *args: ArgsT.args, **kwargs: ArgsT.kwargs):
        return self.host.spawn(
            self.func,
            self.options,
            args,
            kwargs,
            application_name=self.app_name,
            application_auth_mode=self.app_auth,
        )

    def __call__(self, *args: ArgsT.args, **kwargs: ArgsT.kwargs) -> ReturnT:
        try:
            return self.host.run(
                self.func,
                self.options,
                args=args,
                kwargs=kwargs,
                application_name=self.app_name,
                application_auth_mode=self.app_auth,
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

    def run_local(self, *args: ArgsT.args, **kwargs: ArgsT.kwargs) -> ReturnT:
        import asyncio  # noqa: PLC0415
        import inspect  # noqa: PLC0415
        import os  # noqa: PLC0415
        from typing import Awaitable, cast  # noqa: PLC0415

        func = self.func
        previous_isolate_env = os.environ.get("IS_ISOLATE_AGENT")
        os.environ["IS_ISOLATE_AGENT"] = "1"
        try:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                awaited = cast(Awaitable[ReturnT], result)

                async def _await_result():
                    return await awaited

                return asyncio.run(_await_result())  # type: ignore[return-value]
            return result
        finally:
            if previous_isolate_env is None:
                del os.environ["IS_ISOLATE_AGENT"]
            else:
                os.environ["IS_ISOLATE_AGENT"] = previous_isolate_env

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
            serve_func = ServeWrapper(self.raw_func)
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
