from __future__ import annotations

import asyncio
import inspect
import os
import queue
import re
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional

import fastapi
import grpc.aio as async_grpc

from fal._serialization import include_modules_from
from fal._typing import EndpointT
from fal.api import (
    SERVE_REQUIREMENTS,
    BaseServable,
    IsolatedFunction,
    RouteSignature,
    SpawnInfo,
)
from fal.api import (
    function as fal_function,
)
from fal.auth import key_credentials
from fal.container import ContainerImage
from fal.exceptions import FalServerlessException, RequestCancelledException
from fal.logging import get_logger
from fal.realtime import realtime  # noqa: F401
from fal.ref import set_current_app
from fal.sdk import (
    ApplicationHealthCheckConfig,
    AuthModeLiteral,
    HealthCheck,
    RetryConditionLiteral,
)
from fal.toolkit.file import request_lifecycle_preference

REALTIME_APP_REQUIREMENTS = ["websockets", "msgpack"]
REQUEST_ID_KEY = "x-fal-request-id"
REQUEST_ENDPOINT_KEY = "x-fal-endpoint"
DEFAULT_APP_FILES_IGNORE = [
    r"\.pyc$",
    r"__pycache__/",
    r"\.git/",
    r"\.DS_Store$",
]


logger = get_logger(__name__)


async def _call_any_fn(fn, *args, **kwargs):
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    else:
        return fn(*args, **kwargs)


async def open_isolate_channel(address: str) -> async_grpc.Channel | None:
    channel = async_grpc.insecure_channel(
        address,
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
            ("grpc.min_reconnect_backoff_ms", 0),
            ("grpc.max_reconnect_backoff_ms", 100),
            ("grpc.dns_min_time_between_resolutions_ms", 100),
        ],
    )

    try:
        channel_status = channel.channel_ready()

        await asyncio.wait_for(channel_status, timeout=1)
    except asyncio.TimeoutError:
        await channel.close(None)
        print("[DEBUG] Timed out trying to connect to local isolate")
        return None

    return channel


async def _set_logger_labels(
    logger_labels: dict[str, str], channel: async_grpc.Channel
):
    try:
        import sys

        from isolate.server import definitions

        # Flush any prints that were buffered before setting the logger labels
        sys.stderr.flush()
        sys.stdout.flush()

        isolate = definitions.IsolateStub(channel)
        isolate_request = definitions.SetMetadataRequest(
            # TODO: when submit is shipped, get task_id from an env var
            task_id="RUN",
            metadata=definitions.TaskMetadata(logger_labels=logger_labels),
        )
        res = isolate.SetMetadata(isolate_request)
        code = await res.code()
        assert str(code) == "StatusCode.OK", str(code)
    except BaseException:
        # ignore if shutting down
        if os.environ.get("FAL_RUNNER_STATE") == "TERMINATING":
            return

        logger.debug("Failed to set logger labels", exc_info=True)


def wrap_app(cls: type[App], **kwargs) -> IsolatedFunction:
    include_modules_from(cls)

    host = kwargs.get("host", None)
    if host:
        cls.local_file_path = host.local_file_path

    def initialize_and_serve():
        import threading

        app = cls()
        set_current_app(app)

        if threading.current_thread() == threading.main_thread():
            return app.serve()
        else:
            asyncio.set_event_loop(asyncio.new_event_loop())
            asyncio.run(app.serve())

    # if the function is not marked with _run_on_main_thread, it runs on a thread pool
    # however in thread pool, the function cannot receive SIGTERM
    # we run the function on main thread so SIGTERM can be propagated to the app
    initialize_and_serve._run_on_main_thread = True  # type: ignore[attr-defined]

    metadata = {}
    app = cls(_allow_init=True)

    metadata["openapi"] = app.openapi()

    routes = app.collect_routes()
    initialize_and_serve._routes = [r.path for r in routes.keys()] or ["/"]  # type: ignore[attr-defined]
    realtime_app = any(route.is_websocket for route in routes)

    kind = cls.host_kwargs.pop("kind", "virtualenv")

    wrapper = fal_function(
        kind,
        requirements=cls.requirements,
        local_python_modules=cls.local_python_modules,
        machine_type=cls.machine_type,
        num_gpus=cls.num_gpus,
        regions=cls.regions,
        **cls.host_kwargs,
        **kwargs,
        metadata=metadata,
        exposed_port=8080,
        serve=False,
    )
    fn = wrapper(initialize_and_serve)
    fn.options.add_requirements(SERVE_REQUIREMENTS)
    if realtime_app:
        fn.options.add_requirements(REALTIME_APP_REQUIREMENTS)

    fn.app_name = cls.app_name
    fn.app_auth = cls.app_auth

    return fn


@dataclass
class AppClientError(FalServerlessException):
    message: str
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)


class AppSpawnInfo:
    def __init__(self, info: SpawnInfo):
        self.info = info

    @property
    def url(self):
        return self.info.url

    @property
    def application(self):
        return self.info.application

    @property
    def logs(self):
        return self.info.logs

    @property
    def stream(self):
        return self.info.stream

    @property
    def future(self):
        return self.info.future

    def wait(
        self,
        *,
        health_request_timeout: int = 30,
        startup_timeout: int = 60,
        health_check_interval: float = 0.5,
        headers: dict[str, str] | None = None,
    ) -> None:
        import httpx

        url = self.url
        if url is None:
            raise AppClientError(
                "App spawn failed: no URL returned",
                500,
                {},
            )

        start_time = time.perf_counter()
        health_url = url + "/health"
        last_error = None
        attempt = 0
        request_headers = _default_auth_headers() if headers is None else headers

        with httpx.Client() as client:
            while time.perf_counter() - start_time < startup_timeout:
                attempt += 1

                try:
                    resp = client.get(
                        health_url,
                        timeout=health_request_timeout,
                        headers=request_headers,
                    )
                except httpx.TimeoutException:
                    last_error = (
                        f"Request timed out after {health_request_timeout} seconds"
                    )
                except httpx.TransportError as e:
                    last_error = f"Network error: {e}"
                else:
                    if resp.is_success:
                        return

                    if resp.status_code in (500, 404):
                        last_error = f"Server not ready (HTTP {resp.status_code})"
                    else:
                        raise AppClientError(
                            "Health check failed with non-retryable error: "
                            f"{resp.status_code} {resp.text}",
                            resp.status_code,
                            dict(resp.headers),
                        )

                time.sleep(health_check_interval)

        raise AppClientError(
            f"Health check failed after {startup_timeout}s "
            f"({attempt} attempts). Last error: {last_error}",
            500,
            {},
        )


def _default_auth_headers() -> dict[str, str]:
    key_creds = key_credentials()
    if not key_creds:
        return {}
    key_id, key_secret = key_creds
    return {"Authorization": f"Key {key_id}:{key_secret}"}


class EndpointClient:
    def __init__(
        self,
        url,
        endpoint,
        signature,
        timeout: int | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.url = url
        self.endpoint = endpoint
        self.signature = signature
        self.timeout = timeout
        self.headers = headers or {}

        annotations = endpoint.__annotations__ or {}
        self.return_type = annotations.get("return") or None

    def __call__(self, data):
        import httpx

        with httpx.Client() as client:
            url = self.url + self.signature.path
            resp = client.post(
                self.url + self.signature.path,
                json=data.dict() if hasattr(data, "dict") else dict(data),
                timeout=self.timeout,
                headers=self.headers,
            )
            if not resp.is_success:
                # allow logs to be printed before raising the exception
                time.sleep(1)
                raise AppClientError(
                    f"Failed to POST {url}: {resp.status_code} {resp.text}",
                    status_code=resp.status_code,
                    headers=resp.headers,
                )
            resp_dict = resp.json()

        if not self.return_type:
            return resp_dict

        return self.return_type(**resp_dict)


class AppClient:
    def __init__(
        self,
        cls,
        url,
        timeout: int | None = None,
    ):
        self.url = url
        self.cls = cls
        self._headers = _default_auth_headers()

        user_name, app_name = url.rstrip("/").rsplit("/", 2)[-2:]
        self.application = f"{user_name}/{app_name}"

        for name, endpoint in inspect.getmembers(cls, inspect.isfunction):
            signature = getattr(endpoint, "route_signature", None)
            if signature is None:
                continue
            endpoint_client = EndpointClient(
                self.url,
                endpoint,
                signature,
                timeout=timeout,
                headers=self._headers,
            )
            setattr(self, name, endpoint_client)

    @classmethod
    @contextmanager
    def connect(
        cls,
        app_cls,
        *,
        health_request_timeout: int = 30,
        startup_timeout: int = 60,
        health_check_interval: float = 0.5,
    ):
        info = app_cls.spawn()
        _shutdown_event = threading.Event()

        def _print_logs():
            while not _shutdown_event.is_set():
                try:
                    log = info.logs.get(timeout=0.1)
                except queue.Empty:
                    continue
                print(log)

        _log_printer = threading.Thread(target=_print_logs, daemon=True)
        _log_printer.start()

        try:
            info.wait(
                health_request_timeout=health_request_timeout,
                startup_timeout=startup_timeout,
                health_check_interval=health_check_interval,
            )

            client = cls(app_cls, info.url)
            yield client
        finally:
            info.stream.cancel()
            _shutdown_event.set()
            _log_printer.join()

    def health(self):
        import httpx

        with httpx.Client() as client:
            resp = client.get(self.url + "/health", headers=self._headers)
            resp.raise_for_status()
            return resp.json()


PART_FINDER_RE = re.compile(r"[A-Z][a-z]*")


def _to_fal_app_name(name: str) -> str:
    # Existing behavior (unchanged) - Convert PascalCase to kebab-case
    # e.g., MyGoodApp -> my-good-app, ONNXModel -> o-n-n-x-model
    result = "-".join(part.lower() for part in PART_FINDER_RE.findall(name))

    # If existing behavior worked, return it (backwards compatible)
    if result:
        return result

    # Fallback for snake_case (mock_model -> mock-model)
    if "_" in name:
        return "-".join(part.lower() for part in name.split("_") if part)

    # Ultimate fallback: just lowercase the name
    if name:
        return name.lower()

    # This should never happen, but provide a clear error if it does
    raise ValueError(
        f"Cannot derive app name from '{name}'. "
        f"Please use --app-name to specify a name explicitly."
    )


def _print_python_packages() -> None:
    from importlib.metadata import distributions

    packages = [f"{dist.metadata['Name']}=={dist.version}" for dist in distributions()]

    print("[debug] Python packages installed:", ", ".join(packages))


def _include_app_files_path(
    local_file_path: str | None, app_files_context_dir: str | None
):
    base_cloud_dir = Path("/app")
    if local_file_path is None:
        return

    # In case of container apps, the /app directory is not created by default
    # so we need to check if it exists before proceeding
    if not base_cloud_dir.exists():
        return

    base_path = Path(local_file_path).resolve()
    if base_path.is_dir():
        original_script_dir = base_path
    else:
        original_script_dir = base_path.parent

    if app_files_context_dir:
        context_path = Path(app_files_context_dir)
        if context_path.is_absolute():
            final_script_dir = context_path.resolve()
        else:
            final_script_dir = (original_script_dir / context_path).resolve()

        # relative path between the original script dir
        # and where the app_files_context_dir is targetting
        relative_path = os.path.relpath(original_script_dir, final_script_dir)
        # cloud final_path based on the `/app` base dir,
        final_path = base_cloud_dir / Path(relative_path)
    else:
        # if no app_files_context_dir is provided, the base directory is the root
        final_path = base_cloud_dir

    # Create the final path if it doesn't exist
    # This is for cases when fal app is not in root
    # and its parent directory is not in app_files
    # Which means that the relative path to app won't be created by default
    final_path.mkdir(parents=True, exist_ok=True)

    # Add local files deployment path to sys.path so imports
    # work correctly in the isolate agent
    # Append the final path to sys.path first so that the
    # relative directory is resolved first in case of conflicts
    sys.path.append(str(final_path))

    # Add the base cloud dir path to sys.path so that
    # the app can access the files in the top level directory
    # This is for cases when fal app is not in root,
    # and user wants to access the files without using relative imports
    sys.path.append(str(base_cloud_dir))

    # Change the current working directory to the path of the app
    # so that the app can access the files in the current directory
    os.chdir(str(final_path))


@dataclass
class RequestContext:
    request_id: str | None
    endpoint: str | None
    lifecycle_preference: dict[str, str] | None
    headers: fastapi.Header


class App(BaseServable):
    """Create a fal serverless application.

    Subclass this to define your application with custom setup, endpoints,
    and configuration. The App class handles model loading, request routing,
    and lifecycle management.

    Example:
        >>> class TextToImage(fal.App, machine_type="GPU"):
        ...     requirements = ["diffusers", "torch"]
        ...
        ...     def setup(self):
        ...         self.pipe = StableDiffusionPipeline.from_pretrained(
        ...             "runwayml/stable-diffusion-v1-5"
        ...         )
        ...
        ...     @fal.endpoint("/")
        ...     def generate(self, prompt: str) -> dict:
        ...         image = self.pipe(prompt).images[0]
        ...         return {"url": fal.toolkit.upload_image(image)}

    Attributes:
        requirements: List of pip packages to install in the environment.
            Supports standard pip syntax including version specifiers.
            Example: `["numpy==1.24.0", "torch>=2.0.0"]`
        local_python_modules: List of local Python module names to include
            in the deployment. Use for custom code not available on PyPI.
            Example: `["my_utils", "models"]`
        machine_type: Compute instance type for your application. CPU options: 'XS',
            'S' (default), 'M', 'L'. GPU options: 'GPU-A6000', 'GPU-A100', 'GPU-H100',
            'GPU-H200', 'GPU-B200'. Use a string for a single type, or a list to
            define fallback types (tried in order until one is available).
            Example: `"GPU-A100"` or `["GPU-H100", "GPU-A100"]`
        num_gpus: Number of GPUs to allocate. Only applies to GPU machine types.
        regions: Allowed regions for deployment. None means any region.
            Example: `["us-east", "eu-west"]`
        host_kwargs: Advanced configuration dictionary passed to the host.
            For internal use. Prefer using class attributes instead.
        app_name: Custom name for the application. Defaults to class name.
        app_auth: Authentication mode. Options: 'private' (API key required),
            'public' (no auth), 'shared' (shareable link).
        app_files: List of files/directories to include in deployment.
            Example: `["./models", "./config.yaml"]`
        app_files_ignore: Regex patterns to exclude from deployment.
            Default excludes `.pyc`, `__pycache__`, `.git`, `.DS_Store`.
        app_files_context_dir: Base directory for resolving app_files paths.
            Defaults to the directory containing the app file.
        request_timeout: Maximum seconds for a single request. None for default.
        startup_timeout: Maximum seconds for app startup/setup. None for default.
        min_concurrency: Minimum warm instances to keep running. Set to 1+ to
            avoid cold starts. Default is 0 (scale to zero).
        max_concurrency: Maximum instances to scale up to.
        concurrency_buffer: Additional instances to keep warm above current load.
        concurrency_buffer_perc: Percentage buffer of instances above current load.
        scaling_delay: Seconds to wait for a request to be picked up by a runner
            before triggering a scale up. Useful for apps with slow startup times.
        max_multiplexing: Maximum concurrent requests per instance.
        kind: Deployment kind. For internal use.
        image: Custom container image for the application. Use ContainerImage
            to specify a Dockerfile.
    """

    requirements: ClassVar[list[str]] = []
    local_python_modules: ClassVar[list[str]] = []
    machine_type: ClassVar[str | list[str]] = "S"
    num_gpus: ClassVar[int | None] = None
    regions: ClassVar[Optional[list[str]]] = None
    host_kwargs: ClassVar[dict[str, Any]] = {
        "_scheduler": "nomad",
        "_scheduler_options": {
            "storage_region": "us-east",
        },
        "resolver": "uv",
        "keep_alive": 60,
    }
    app_name: ClassVar[Optional[str]] = None
    app_auth: ClassVar[Optional[AuthModeLiteral]] = None
    app_files: ClassVar[list[str]] = []
    app_files_ignore: ClassVar[list[str]] = DEFAULT_APP_FILES_IGNORE
    app_files_context_dir: ClassVar[Optional[str]] = None
    request_timeout: ClassVar[Optional[int]] = None
    startup_timeout: ClassVar[Optional[int]] = None
    min_concurrency: ClassVar[Optional[int]] = None
    max_concurrency: ClassVar[Optional[int]] = None
    concurrency_buffer: ClassVar[Optional[int]] = None
    concurrency_buffer_perc: ClassVar[Optional[int]] = None
    scaling_delay: ClassVar[Optional[int]] = None
    max_multiplexing: ClassVar[Optional[int]] = None
    kind: ClassVar[Optional[str]] = None
    image: ClassVar[Optional[ContainerImage]] = None
    local_file_path: ClassVar[Optional[str]] = None
    skip_retry_conditions: ClassVar[Optional[list[RetryConditionLiteral]]] = None
    termination_grace_period_seconds: ClassVar[Optional[int]] = None

    isolate_channel: async_grpc.Channel | None = None

    _current_request_context: ContextVar[RequestContext] | None = None

    def __init_subclass__(cls, **kwargs):
        app_name = kwargs.pop("name", None) or _to_fal_app_name(cls.__name__)
        parent_settings = getattr(cls, "host_kwargs", {})
        cls.host_kwargs = {**parent_settings, **kwargs}

        for key in parent_settings.keys():
            val = getattr(cls, key, None)
            if val is not None:
                cls.host_kwargs[key] = val

        if cls.request_timeout is not None:
            cls.host_kwargs["request_timeout"] = cls.request_timeout

        if cls.startup_timeout is not None:
            cls.host_kwargs["startup_timeout"] = cls.startup_timeout

        if cls.app_files:
            cls.host_kwargs["app_files"] = cls.app_files

        if cls.app_files_ignore:
            cls.host_kwargs["app_files_ignore"] = cls.app_files_ignore

        if cls.app_files_context_dir is not None:
            cls.host_kwargs["app_files_context_dir"] = cls.app_files_context_dir
            if not cls.app_files:
                raise ValueError(
                    "app_files_context_dir is only supported when app_files is provided"
                )

        if cls.min_concurrency is not None:
            cls.host_kwargs["min_concurrency"] = cls.min_concurrency

        if cls.max_concurrency is not None:
            cls.host_kwargs["max_concurrency"] = cls.max_concurrency

        if cls.concurrency_buffer is not None:
            cls.host_kwargs["concurrency_buffer"] = cls.concurrency_buffer

        if cls.concurrency_buffer_perc is not None:
            cls.host_kwargs["concurrency_buffer_perc"] = cls.concurrency_buffer_perc

        if cls.scaling_delay is not None:
            cls.host_kwargs["scaling_delay"] = cls.scaling_delay

        if cls.max_multiplexing is not None:
            cls.host_kwargs["max_multiplexing"] = cls.max_multiplexing

        if cls.kind is not None:
            cls.host_kwargs["kind"] = cls.kind

        if cls.image is not None:
            cls.host_kwargs["image"] = cls.image
            cls.host_kwargs["kind"] = "container"
            # For consistency, check also here (same check with function decorator)
            if cls.app_files:
                raise ValueError("app_files is not supported for container apps.")

        if cls.skip_retry_conditions is not None:
            cls.host_kwargs["skip_retry_conditions"] = cls.skip_retry_conditions

        if cls.termination_grace_period_seconds is not None:
            cls.host_kwargs["termination_grace_period_seconds"] = (
                cls.termination_grace_period_seconds
            )

        cls.host_kwargs["health_check_config"] = cls.get_health_check_config()

        cls.app_name = getattr(cls, "app_name") or app_name

        if cls.__init__ is not App.__init__:
            raise ValueError(
                "App classes should not override __init__ directly. "
                "Use setup() instead."
            )

        if cls.requirements and cls.host_kwargs.get("kind") == "container":
            from fal.console import console

            console.print(
                "\n[yellow]WARNING:[/yellow] Using [bold]requirements[/bold] with "
                "container apps is not recommended. For better performance, "
                "install dependencies in the Dockerfile instead.\n"
            )

    def __init__(self, *, _allow_init: bool = False):
        if not _allow_init and not os.getenv("IS_ISOLATE_AGENT"):
            cls_name = self.__class__.__name__
            raise NotImplementedError(
                "Running apps through SDK is not implemented yet. "
                f"Please use `fal run path/to/app.py::{cls_name}` to run your app."
            )

    def __getstate__(self) -> dict[str, Any]:
        # we might need to pickle the app sometimes,
        # e.g. in fal distributed workers from our toolkit
        state = self.__dict__.copy()
        state.pop("_current_request_context", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._current_request_context = None

    @classmethod
    def get_endpoints(cls) -> list[str]:
        return [
            signature.path
            for _, endpoint in inspect.getmembers(cls, inspect.isfunction)
            if (signature := getattr(endpoint, "route_signature", None))
        ]

    @classmethod
    def spawn(cls) -> AppSpawnInfo:
        # import wrap_app explicitly to avoid reference to wrap_app during pickling
        from fal.app import wrap_app

        app = wrap_app(cls)
        return AppSpawnInfo(app.spawn())

    @classmethod
    def get_health_check_config(cls) -> Optional[ApplicationHealthCheckConfig]:
        health_check_path: str | None = None
        health_check: HealthCheck | None = None

        for _, endpoint in inspect.getmembers(cls, inspect.isfunction):
            signature = getattr(endpoint, "route_signature", None)
            if not signature or not signature.health_check:
                continue

            if signature.is_websocket:
                raise ValueError(
                    "Health check endpoints cannot be websocket endpoints."
                )

            if health_check is not None:
                raise ValueError(
                    "Multiple health check endpoints found. "
                    "An app can only have one health check endpoint."
                )

            health_check_path = signature.path
            health_check = signature.health_check

        if health_check is None or health_check_path is None:
            return None

        return ApplicationHealthCheckConfig(
            path=health_check_path,
            start_period_seconds=health_check.start_period_seconds,
            timeout_seconds=health_check.timeout_seconds,
            failure_threshold=health_check.failure_threshold,
            call_regularly=health_check.call_regularly,
        )

    def collect_routes(self) -> dict[RouteSignature, Callable[..., Any]]:
        return {
            signature: endpoint
            for _, endpoint in inspect.getmembers(self, inspect.ismethod)
            if (signature := getattr(endpoint, "route_signature", None))
        }

    @asynccontextmanager
    async def lifespan(self, app: fastapi.FastAPI):
        os.environ["FAL_RUNNER_STATE"] = "SETUP"

        # Configure sys.path based on deployment type:
        # - app_files: files synced to /app
        # - container: files baked into image
        self._current_request_context = ContextVar(
            "_current_request_context",
            default=RequestContext(
                request_id=None, endpoint=None, lifecycle_preference=None, headers={}
            ),
        )

        # We want to not do any directory changes for container apps,
        # since we don't have explicit checks to see the kind of app
        # We check for app_files here and check kind and app_files earlier
        # to ensure that container apps don't have app_files
        if self.app_files:
            # For app_files deployments (always use /app)
            _include_app_files_path(self.local_file_path, self.app_files_context_dir)
        elif self.image is not None:
            # For containers, add the working directory to sys.path
            # isolate's runpy.run_path() overrides sys.path[0],
            # so the working directory is never added to sys.path
            sys.path.insert(0, "")
        _print_python_packages()
        await _call_any_fn(self.setup)

        os.environ["FAL_RUNNER_STATE"] = "RUNNING"

        try:
            yield
        finally:
            os.environ["FAL_RUNNER_STATE"] = "TERMINATING"
            await _call_any_fn(self.teardown)

    @property
    def current_request(self) -> RequestContext | None:
        if self._current_request_context is None:
            return None
        return self._current_request_context.get()

    def health(self):
        return {"version": self.version}

    def setup(self):
        """Setup the application before serving."""

    def handle_exit(self):
        """Handle exit signal."""

    def teardown(self):
        """Teardown the application after serving."""

    def _add_extra_middlewares(self, app: fastapi.FastAPI):
        @app.middleware("http")
        async def provide_hints_headers(request, call_next):
            response = await call_next(request)
            try:
                # make sure the hints can be encoded in latin-1, so we don't crash
                # when serving.
                # https://github.com/encode/starlette/blob/a766a58d14007f07c0b5782fa78cdc370b892796/starlette/datastructures.py#L568
                hints = []
                for hint in self.provide_hints():
                    try:
                        _ = hint.encode("latin-1")
                        hints.append(hint)
                    except UnicodeEncodeError:
                        from fastapi.logger import logger

                        logger.warning(
                            "Ignoring hint %s for %s because it can't be encoded in "
                            "latin-1",
                            hint,
                            self.__class__.__name__,
                        )

                response.headers["X-Fal-Runner-Hints"] = ",".join(hints)
            except NotImplementedError:
                # This lets us differentiate between apps that don't provide hints
                # and apps that provide empty hints.
                pass
            except Exception:
                from fastapi.logger import logger

                logger.exception(
                    "Failed to provide hints for %s",
                    self.__class__.__name__,
                )
            return response

        @app.middleware("http")
        async def set_current_request_context(request, call_next):
            if self._current_request_context is None:
                from fastapi.logger import logger

                logger.warning(
                    "request context is not set. "
                    "lifespan may not have worked as expected."
                )
                return await call_next(request)

            context = RequestContext(
                request_id=request.headers.get(REQUEST_ID_KEY),
                endpoint=request.headers.get(REQUEST_ENDPOINT_KEY),
                lifecycle_preference=request_lifecycle_preference(request),
                headers=request.headers,
            )

            token = self._current_request_context.set(context)
            try:
                return await call_next(request)
            finally:
                self._current_request_context.reset(token)

        @app.middleware("http")
        async def set_request_id(request, call_next):
            # NOTE: Setting request_id is not supported for websocket/realtime endpoints
            if not os.getenv("IS_ISOLATE_AGENT") or not os.environ.get(
                "NOMAD_ALLOC_PORT_grpc"
            ):
                # If not running in the expected environment, skip setting request_id
                return await call_next(request)

            if self.isolate_channel is None:
                grpc_port = os.environ.get("NOMAD_ALLOC_PORT_grpc")
                self.isolate_channel = await open_isolate_channel(
                    f"localhost:{grpc_port}"
                )

            if self.isolate_channel is None:
                return await call_next(request)

            request_id = request.headers.get(REQUEST_ID_KEY)
            request_endpoint = request.headers.get(REQUEST_ENDPOINT_KEY)

            if request_id is None and request_endpoint is None:
                return await call_next(request)

            labels_to_set = {}
            if request_id:
                labels_to_set["fal_request_id"] = request_id
            if request_endpoint:
                labels_to_set["fal_endpoint"] = request_endpoint

            await _set_logger_labels(labels_to_set, channel=self.isolate_channel)

            async def _unset_at_end():
                await _set_logger_labels({}, channel=self.isolate_channel)  # type: ignore

            try:
                response: fastapi.responses.Response = await call_next(request)
            except BaseException:
                await _unset_at_end()
                raise
            else:
                # We need to wait for the entire response to be sent before
                # we can set the logger labels back to the default.
                background_tasks = fastapi.BackgroundTasks()
                background_tasks.add_task(_unset_at_end)
                if response.background:
                    # We normally have no background tasks, but we should handle it
                    background_tasks.add_task(response.background)
                response.background = background_tasks

                return response

        @app.exception_handler(RequestCancelledException)
        async def value_error_exception_handler(
            request, exc: RequestCancelledException
        ):
            from fastapi.responses import JSONResponse

            # A 499 status code is not an officially recognized HTTP status code,
            # but it is sometimes used by servers to indicate that a client has closed
            # the connection without receiving a response
            return JSONResponse({"detail": str(exc)}, 499)

    def _add_extra_routes(self, app: fastapi.FastAPI):
        # TODO remove this once we have a proper health check endpoint
        @app.get("/health")
        def health():
            return self.health()

    def provide_hints(self) -> list[str]:
        """Provide hints for routing the application."""
        raise NotImplementedError


def endpoint(
    path: str,
    *,
    is_websocket: bool = False,
    health_check: HealthCheck | None = None,
) -> Callable[[EndpointT], EndpointT]:
    """Designate the decorated function as an application endpoint."""

    def marker_fn(callable: EndpointT) -> EndpointT:
        if hasattr(callable, "route_signature"):
            raise ValueError(
                f"Can't set multiple routes for the same function: {callable.__name__}"
            )

        callable.route_signature = RouteSignature(  # type: ignore
            path=path,
            is_websocket=is_websocket,
            health_check=health_check,
        )
        return callable

    return marker_fn
