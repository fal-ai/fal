from __future__ import annotations

import inspect
import sys
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial, wraps
from os import PathLike
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    Literal,
    TypeVar,
    cast,
    overload,
)

import dill
import dill.detect
import fal_serverless.flags as flags
import grpc
import isolate
import yaml
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
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

if sys.version_info >= (3, 11):
    from typing import Concatenate
else:
    from typing_extensions import Concatenate

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec


dill.settings["recurse"] = True

ArgsT = ParamSpec("ArgsT")
ReturnT = TypeVar("ReturnT", covariant=True)

BasicConfig = Dict[str, Any]
_UNSET = object()


@dataclass
class FalServerlessError(Exception):
    message: str


@dataclass
class InternalFalServerlessError(Exception):
    message: str


@dataclass
class FalMissingDependencyError(FalServerlessError):
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


def _execution_controller(
    func: Callable[ArgsT, ReturnT],
    *args: ArgsT.args,
    **kwargs: ArgsT.kwargs,
) -> Callable[ArgsT, ReturnT]:
    """Handle the execution of the given user function."""

    @wraps(func)
    def wrapper(*remote_args: ArgsT.args, **remote_kwargs: ArgsT.kwargs) -> ReturnT:
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


FAL_SERVERLESS_DEFAULT_URL = flags.GRPC_HOST
FAL_SERVERLESS_DEFAULT_MACHINE_TYPE = "XS"


import threading


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
    if env["kind"] != "virtualenv":
        return

    used_modules = defaultdict(list)
    scope = {**dill.detect.globalvars(func, recurse=True), **dill.detect.freevars(func)}  # type: ignore

    for name, obj in scope.items():
        if isinstance(obj, IsolatedFunction):
            used_modules["fal_serverless"].append(name)
            continue

        module = inspect.getmodule(obj)
        possible_package = getattr(module, "__package__", None)
        if possible_package:
            pkg_name, *_ = possible_package.split(".")  # type: ignore
        else:
            pkg_name = module.__name__  # type: ignore

        used_modules[canonicalize_name(pkg_name)].append(name)  # type: ignore

    raw_requirements = env.get("requirements", [])
    specified_requirements = set()
    for raw_requirement in raw_requirements:
        requirement = Requirement(raw_requirement)
        specified_requirements.add(canonicalize_name(requirement.name))

    for module_name, used_names in used_modules.items():
        if module_name in specified_requirements:
            continue
        yield module_name, used_names


# TODO: Should we build all these in fal/dbt-fal packages instead?
@dataclass
class FalServerlessHost(Host):
    _SUPPORTED_KEYS = frozenset(
        {"machine_type", "keep_alive", "setup_function", "_base_image"}
    )

    url: str = FAL_SERVERLESS_DEFAULT_URL
    credentials: Credentials = field(default_factory=get_default_credentials)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    _log_printer = IsolateLogPrinter(debug=flags.DEBUG)

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
        func: Callable[ArgsT, ReturnT],
        options: Options,
        application_name: str | None = None,
        application_auth_mode: Literal["public", "shared", "private"] | None = None,
    ) -> str | None:
        environment_options = options.environment.copy()
        environment_options.setdefault("python_version", active_python())
        environments = [self._connection.define_environment(**environment_options)]

        machine_type = options.host.get(
            "machine_type", FAL_SERVERLESS_DEFAULT_MACHINE_TYPE
        )
        keep_alive = options.host.get("keep_alive", FAL_SERVERLESS_DEFAULT_KEEP_ALIVE)
        base_image = options.host.get("_base_image", None)

        machine_requirements = MachineRequirements(
            machine_type=machine_type,
            keep_alive=keep_alive,
            base_image=base_image,
        )

        partial_func = _execution_controller(func)

        for partial_result in self._connection.register(
            partial_func,
            environments,
            application_name=application_name,
            application_auth_mode=application_auth_mode,
            machine_requirements=machine_requirements,
        ):
            for log in partial_result.logs:
                self._log_printer.print(log)

            if partial_result.result:
                return partial_result.result.application_id

    @_handle_grpc_error()
    def schedule(
        self,
        func: Callable[ArgsT, ReturnT],
        cron: str,
        options: Options,
    ) -> str | None:
        application_id = self.register(func, options)
        if application_id is None:
            return None
        return self._connection.schedule_cronjob(application_id, cron)

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
        base_image = options.host.get("_base_image", None)
        setup_function = options.host.get("setup_function", None)

        machine_requirements = MachineRequirements(
            machine_type=machine_type,
            keep_alive=keep_alive,
            base_image=base_image,
        )

        return_value = _UNSET
        # Allow isolate provided arguments (such as setup function) to take
        # precedence over the ones provided by the user.
        partial_func = _execution_controller(func, *args, **kwargs)
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


# Overload @isolated to help users identify the correct signature.
# NOTE: This is both in sync with host options and with environment configs from `isolate` package.


## virtualenv
### LocalHost
@overload
def isolated(
    kind: Literal["virtualenv"] = "virtualenv",
    *,
    python_version: str | None = None,
    requirements: list[str] | None = None,
    # Common options
    host: LocalHost,
    serve: Literal[False] = False,
    exposed_port: int | None = None,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], IsolatedFunction[ArgsT, ReturnT]
]:
    ...


@overload
def isolated(
    kind: Literal["virtualenv"] = "virtualenv",
    *,
    python_version: str | None = None,
    requirements: list[str] | None = None,
    # Common options
    host: LocalHost,
    serve: Literal[True],
    exposed_port: int | None = None,
) -> Callable[[Callable[Concatenate[ArgsT], ReturnT]], IsolatedFunction[[], None]]:
    ...


### FalServerlessHost
@overload
def isolated(
    kind: Literal["virtualenv"] = "virtualenv",
    *,
    python_version: str | None = None,
    requirements: list[str] | None = None,
    # Common options
    host: FalServerlessHost = _DEFAULT_HOST,
    serve: Literal[False] = False,
    exposed_port: int | None = None,
    # FalServerlessHost options
    machine_type: str = FAL_SERVERLESS_DEFAULT_MACHINE_TYPE,
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    _base_image: str | None = None,
    setup_function: Callable[..., None] | None = None,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], IsolatedFunction[ArgsT, ReturnT]
]:
    ...


@overload
def isolated(
    kind: Literal["virtualenv"] = "virtualenv",
    *,
    python_version: str | None = None,
    requirements: list[str] | None = None,
    # Common options
    host: FalServerlessHost = _DEFAULT_HOST,
    serve: Literal[True],
    exposed_port: int | None = None,
    # FalServerlessHost options
    machine_type: str = FAL_SERVERLESS_DEFAULT_MACHINE_TYPE,
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    _base_image: str | None = None,
    setup_function: Callable[..., None] | None = None,
) -> Callable[[Callable[Concatenate[ArgsT], ReturnT]], IsolatedFunction[[], None]]:
    ...


## conda
### LocalHost
@overload
def isolated(
    kind: Literal["conda"],
    *,
    python_version: str | None = None,
    env_dict: dict[str, str] | None = None,
    env_yml: PathLike | str | None = None,
    env_yml_str: str | None = None,
    packages: list[str] | None = None,
    pip: list[str] | None = None,
    channels: list[str] | None = None,
    # Common options
    host: LocalHost,
    serve: Literal[False] = False,
    exposed_port: int | None = None,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], IsolatedFunction[ArgsT, ReturnT]
]:
    ...


@overload
def isolated(
    kind: Literal["conda"],
    *,
    python_version: str | None = None,
    env_dict: dict[str, str] | None = None,
    env_yml: PathLike | str | None = None,
    env_yml_str: str | None = None,
    packages: list[str] | None = None,
    pip: list[str] | None = None,
    channels: list[str] | None = None,
    # Common options
    host: LocalHost,
    serve: Literal[True],
    exposed_port: int | None = None,
) -> Callable[[Callable[Concatenate[ArgsT], ReturnT]], IsolatedFunction[[], None]]:
    ...


### FalServerlessHost
@overload
def isolated(
    kind: Literal["conda"],
    *,
    python_version: str | None = None,
    env_dict: dict[str, str] | None = None,
    env_yml: PathLike | str | None = None,
    env_yml_str: str | None = None,
    packages: list[str] | None = None,
    pip: list[str] | None = None,
    channels: list[str] | None = None,
    # Common options
    host: FalServerlessHost = _DEFAULT_HOST,
    serve: Literal[False] = False,
    exposed_port: int | None = None,
    # FalServerlessHost options
    machine_type: str = FAL_SERVERLESS_DEFAULT_MACHINE_TYPE,
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    _base_image: str | None = None,
    setup_function: Callable[..., None] | None = None,
) -> Callable[
    [Callable[Concatenate[ArgsT], ReturnT]], IsolatedFunction[ArgsT, ReturnT]
]:
    ...


@overload
def isolated(
    kind: Literal["conda"],
    *,
    python_version: str | None = None,
    env_dict: dict[str, str] | None = None,
    env_yml: PathLike | str | None = None,
    env_yml_str: str | None = None,
    packages: list[str] | None = None,
    pip: list[str] | None = None,
    channels: list[str] | None = None,
    # Common options
    host: FalServerlessHost = _DEFAULT_HOST,
    serve: Literal[True],
    exposed_port: int | None = None,
    # FalServerlessHost options
    machine_type: str = FAL_SERVERLESS_DEFAULT_MACHINE_TYPE,
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE,
    _base_image: str | None = None,
    setup_function: Callable[..., None] | None = None,
) -> Callable[[Callable[Concatenate[ArgsT], ReturnT]], IsolatedFunction[[], None]]:
    ...


# implementation
def isolated(  # type: ignore
    kind: str = "virtualenv",
    *,
    host: Host = _DEFAULT_HOST,
    **config: Any,
):
    options = host.parse_options(kind=kind, **config)

    def wrapper(func: Callable[ArgsT, ReturnT]):
        # wrap it with flask if the serve option is set
        res = templated_flask(func) if options.gateway.get("serve") else func
        proxy = IsolatedFunction(
            host=host,
            func=res,  # type: ignore
            options=options,
        )
        return wraps(res)(proxy)  # type: ignore

    return wrapper


def templated_flask(func: Callable[ArgsT, ReturnT]) -> Callable[[], None]:
    param_names = inspect.signature(func).parameters.keys()

    def templated_flask_wrapper():
        from flask import Flask, jsonify, request
        from flask_cors import CORS

        app = Flask("fal")
        cors = CORS(app, resources={r"/*": {"origins": "*"}})

        @app.route("/", methods=["POST"])
        def flask():
            try:
                body = request.get_json()
                if not isinstance(body, dict):
                    raise TypeError("Body must be a JSON object")

                res: ReturnT = func(**body)  # type: ignore
            except TypeError as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                return jsonify({"error": str(e)}), 500

            return jsonify({"result": res})

        app.run(host="0.0.0.0", port=8080)

    return templated_flask_wrapper


@dataclass
class IsolatedFunction(Generic[ArgsT, ReturnT]):
    host: Host
    func: Callable[ArgsT, ReturnT]
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

    def submit(self, *args: ArgsT.args, **kwargs: ArgsT.kwargs) -> Future[ReturnT]:
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
        return future  # type: ignore

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
                        f"\t- {used_modules!r} (accessed through {', '.join(map(repr, references))})"
                    )

                function_name = self.func.__name__
                raise FalServerlessError(
                    f"Couldn't deserialize your function on the remote server. \n\n[Hint] {function_name!r} "
                    f"function uses the following modules which weren't present in the environment definition:\n"
                    + "\n".join(lines)
                ) from None

    def on(
        self, host: Host | None = None, **config: Any
    ) -> IsolatedFunction[ArgsT, ReturnT]:
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
