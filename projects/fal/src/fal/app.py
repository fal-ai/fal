from __future__ import annotations

import asyncio
import inspect
import json
import os
import queue
import re
import sys
import threading
import time
import typing
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional, TypeVar

import fastapi
import grpc.aio as async_grpc
import httpx

from fal._serialization import include_modules_from
from fal.api import (
    SERVE_REQUIREMENTS,
    BaseServable,
    IsolatedFunction,
    RouteSignature,
)
from fal.api import (
    function as fal_function,
)
from fal.container import ContainerImage
from fal.exceptions import FalServerlessException, RequestCancelledException
from fal.logging import get_logger
from fal.ref import set_current_app
from fal.sdk import (
    ApplicationHealthCheckConfig,
    AuthModeLiteral,
    HealthCheck,
    RetryConditionLiteral,
)
from fal.toolkit.file import request_lifecycle_preference
from fal.toolkit.file.providers.fal import LIFECYCLE_PREFERENCE

REALTIME_APP_REQUIREMENTS = ["websockets", "msgpack"]
REQUEST_ID_KEY = "x-fal-request-id"
REQUEST_ENDPOINT_KEY = "x-fal-endpoint"
DEFAULT_APP_FILES_IGNORE = [
    r"\.pyc$",
    r"__pycache__/",
    r"\.git/",
    r"\.DS_Store$",
]


EndpointT = TypeVar("EndpointT", bound=Callable[..., Any])
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

    return fn


@dataclass
class AppClientError(FalServerlessException):
    message: str
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)


class EndpointClient:
    def __init__(self, url, endpoint, signature, timeout: int | None = None):
        self.url = url
        self.endpoint = endpoint
        self.signature = signature
        self.timeout = timeout

        annotations = endpoint.__annotations__ or {}
        self.return_type = annotations.get("return") or None

    def __call__(self, data):
        with httpx.Client() as client:
            url = self.url + self.signature.path
            resp = client.post(
                self.url + self.signature.path,
                json=data.dict() if hasattr(data, "dict") else dict(data),
                timeout=self.timeout,
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

        for name, endpoint in inspect.getmembers(cls, inspect.isfunction):
            signature = getattr(endpoint, "route_signature", None)
            if signature is None:
                continue
            endpoint_client = EndpointClient(
                self.url,
                endpoint,
                signature,
                timeout=timeout,
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
        app = wrap_app(app_cls)
        info = app.spawn()
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
            if info.url is None:
                raise AppClientError(
                    "App spawn failed: no URL returned",
                    status_code=500,
                )

            start_time = time.perf_counter()
            url = info.url + "/health"
            last_error = None
            attempt = 0

            with httpx.Client() as client:
                while time.perf_counter() - start_time < startup_timeout:
                    attempt += 1

                    try:
                        resp = client.get(url, timeout=health_request_timeout)
                    except httpx.TimeoutException:
                        last_error = (
                            f"Request timed out after {health_request_timeout} seconds"
                        )
                    except httpx.TransportError as e:
                        last_error = f"Network error: {e}"
                    else:
                        if resp.is_success:
                            break

                        if resp.status_code in (500, 404):
                            last_error = f"Server not ready (HTTP {resp.status_code})"
                        else:
                            raise AppClientError(
                                "Health check failed with non-retryable error: "
                                f"{resp.status_code} {resp.text}",
                                status_code=resp.status_code,
                                headers=resp.headers,
                            )

                    time.sleep(health_check_interval)
                else:
                    # retry loop completed without success
                    raise AppClientError(
                        f"Health check failed after {startup_timeout}s "
                        f"({attempt} attempts). Last error: {last_error}",
                        status_code=500,
                    )

            client = cls(app_cls, info.url)
            yield client
        finally:
            info.stream.cancel()
            _shutdown_event.set()
            _log_printer.join()

    def health(self):
        with httpx.Client() as client:
            resp = client.get(self.url + "/health")
            resp.raise_for_status()
            return resp.json()


PART_FINDER_RE = re.compile(r"[A-Z][a-z]*")


def _to_fal_app_name(name: str) -> str:
    # Convert MyGoodApp into my-good-app
    return "-".join(part.lower() for part in PART_FINDER_RE.findall(name))


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
    headers: dict[str, str]


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

    isolate_channel: async_grpc.Channel | None = None

    _current_request_context: ContextVar[RequestContext | None] | None = None

    def __init_subclass__(cls, **kwargs):
        app_name = kwargs.pop("name", None) or _to_fal_app_name(cls.__name__)
        parent_settings = getattr(cls, "host_kwargs", {})
        cls.host_kwargs = {**parent_settings, **kwargs}

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

        if cls.skip_retry_conditions is not None:
            cls.host_kwargs["skip_retry_conditions"] = cls.skip_retry_conditions

        cls.host_kwargs["health_check_config"] = cls.get_health_check_config()

        cls.app_name = getattr(cls, "app_name") or app_name

        if cls.__init__ is not App.__init__:
            raise ValueError(
                "App classes should not override __init__ directly. "
                "Use setup() instead."
            )

    def __init__(self, *, _allow_init: bool = False):
        if not _allow_init and not os.getenv("IS_ISOLATE_AGENT"):
            cls_name = self.__class__.__name__
            raise NotImplementedError(
                "Running apps through SDK is not implemented yet. "
                f"Please use `fal run path/to/app.py::{cls_name}` to run your app."
            )

    @classmethod
    def get_endpoints(cls) -> list[str]:
        return [
            signature.path
            for _, endpoint in inspect.getmembers(cls, inspect.isfunction)
            if (signature := getattr(endpoint, "route_signature", None))
        ]

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

        self._current_request_context = ContextVar(
            "_current_request_context", default=RequestContext(headers={})
        )

        # We want to not do any directory changes for container apps,
        # since we don't have explicit checks to see the kind of app
        # We check for app_files here and check kind and app_files earlier
        # to ensure that container apps don't have app_files
        if self.app_files:
            _include_app_files_path(self.local_file_path, self.app_files_context_dir)
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

        multiplexing = self.host_kwargs.get("max_multiplexing") or 1
        if multiplexing == 1:
            # just register the middleware if we are not multiplexing
            @app.middleware("http")
            async def set_global_object_preference(request, call_next):
                try:
                    preference_dict = request_lifecycle_preference(request)
                    if preference_dict is not None:
                        # This will not work properly for apps with multiplexing enabled
                        # we may mix up the preferences between requests
                        LIFECYCLE_PREFERENCE.set(preference_dict)
                except Exception:
                    from fastapi.logger import logger

                    logger.exception(
                        "Failed set a global lifecycle preference %s",
                        self.__class__.__name__,
                    )

                try:
                    return await call_next(request)
                finally:
                    # We may miss the global preference if there are operations
                    # being done in the background that go beyond the request
                    LIFECYCLE_PREFERENCE.set(None)

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

        @app.middleware("http")
        async def set_current_request_context(request, call_next):
            if self._current_request_context is None:
                from fastapi.logger import logger

                logger.warning(
                    "request context is not set. "
                    "lifespan may not have worked as expected."
                )
                return await call_next(request)

            context = RequestContext(headers=dict(request.headers))

            token = self._current_request_context.set(context)
            try:
                return await call_next(request)
            finally:
                self._current_request_context.reset(token)

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


def _fal_websocket_template(
    func: EndpointT,
    route_signature: RouteSignature,
) -> EndpointT:
    # A template for fal's realtime websocket endpoints to basically
    # be a boilerplate for the user to fill in their inference function
    # and start using it.

    import asyncio
    from collections import deque
    from contextlib import suppress

    import msgpack
    from fastapi import WebSocket, WebSocketDisconnect

    async def mirror_input(queue: deque[Any], websocket: WebSocket) -> None:
        while True:
            try:
                raw_input = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=route_signature.session_timeout,
                )
            except asyncio.TimeoutError:
                return

            input = msgpack.unpackb(raw_input, raw=False)
            if route_signature.input_modal:
                input = route_signature.input_modal(**input)

            queue.append(input)

    async def mirror_output(
        self,
        queue: deque[Any],
        websocket: WebSocket,
    ) -> None:
        loop = asyncio.get_event_loop()
        max_allowed_buffering = route_signature.buffering or 1
        outgoing_messages: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=max_allowed_buffering * 2  # x2 for outgoing timings
        )

        async def emit(message):
            if isinstance(message, bytes):
                await websocket.send_bytes(message)
            elif isinstance(message, str):
                await websocket.send_text(message)
            else:
                raise TypeError(f"Can't send message of type {type(message)}")

        async def background_emitter():
            while True:
                output = await outgoing_messages.get()
                await emit(output)
                outgoing_messages.task_done()

        emitter = asyncio.create_task(background_emitter())

        while True:
            if not queue:
                await asyncio.sleep(0.05)
                continue

            input = queue.popleft()
            if input is None or emitter.done():
                if not emitter.done():
                    await outgoing_messages.join()
                    emitter.cancel()

                with suppress(asyncio.CancelledError):
                    await emitter
                return None  # End of input

            batch = [input]
            while queue and len(batch) < route_signature.max_batch_size:
                next_input = queue.popleft()
                if hasattr(input, "can_batch") and not input.can_batch(
                    next_input, len(batch)
                ):
                    queue.appendleft(next_input)
                    break
                batch.append(next_input)

            t0 = loop.time()
            if inspect.iscoroutinefunction(func):
                output = await func(self, *batch)
            else:
                output = await loop.run_in_executor(None, func, self, *batch)  # type: ignore
            total_time = loop.time() - t0
            if not isinstance(output, dict):
                # Handle pydantic output modal
                if hasattr(output, "dict"):
                    output = output.dict()
                else:
                    raise TypeError(
                        "Expected a dict or pydantic model as output, got "
                        f"{type(output)}"
                    )

            messages = [
                msgpack.packb(output, use_bin_type=True),
            ]
            if route_signature.emit_timings:
                # We emit x-fal messages in JSON, no matter what the
                # input/output format is.
                timings = {
                    "type": "x-fal-message",
                    "action": "timings",
                    "timing": total_time,
                }
                messages.append(json.dumps(timings, separators=(",", ":")))

            for message in messages:
                try:
                    outgoing_messages.put_nowait(message)
                except asyncio.QueueFull:
                    await emit(message)

    async def websocket_template(self, websocket: WebSocket) -> None:
        import asyncio

        await websocket.accept()

        queue: deque[Any] = deque(maxlen=route_signature.buffering)
        input_task = asyncio.create_task(mirror_input(queue, websocket))
        input_task.add_done_callback(lambda _: queue.append(None))
        output_task = asyncio.create_task(mirror_output(self, queue, websocket))

        try:
            await asyncio.wait(
                {
                    input_task,
                    output_task,
                },
                return_when=asyncio.FIRST_COMPLETED,
            )
            if input_task.done():
                # User didn't send any input within the timeout
                # so we can just close the connection after the
                # processing of the last input is done.
                input_task.result()
                await asyncio.wait_for(
                    output_task, timeout=route_signature.session_timeout
                )
            else:
                assert output_task.done()

                # The execution of the inference function failed or exitted,
                # so just propagate the result.
                input_task.cancel()
                with suppress(asyncio.CancelledError):
                    await input_task

                output_task.result()
        except WebSocketDisconnect:
            input_task.cancel()
            output_task.cancel()
            with suppress(asyncio.CancelledError):
                await input_task

            with suppress(asyncio.CancelledError):
                await output_task
        except Exception as exc:
            import traceback

            traceback.print_exc()

            await websocket.send_json(
                {
                    "type": "x-fal-error",
                    "error": "INTERNAL_ERROR",
                    "reason": str(exc),
                }
            )
        else:
            await websocket.send_json(
                {
                    "type": "x-fal-error",
                    "error": "TIMEOUT",
                    "reason": "no inputs, reconnect when needed!",
                }
            )

        await websocket.close()

    # Seems like templating + stringified annotations don't play well,
    # so we have to set them manually.
    websocket_template.__annotations__ = {
        "websocket": WebSocket,
        "return": None,
    }
    websocket_template.route_signature = route_signature  # type: ignore
    websocket_template.original_func = func  # type: ignore
    return typing.cast(EndpointT, websocket_template)


_SENTINEL = object()


def realtime(
    path: str,
    *,
    buffering: int | None = None,
    session_timeout: float | None = None,
    input_modal: Any | None = _SENTINEL,
    max_batch_size: int = 1,
) -> Callable[[EndpointT], EndpointT]:
    """Designate the decorated function as a realtime application endpoint."""

    def marker_fn(original_func: EndpointT) -> EndpointT:
        nonlocal input_modal

        if hasattr(original_func, "route_signature"):
            raise ValueError(
                "Can't set multiple routes for the same function: "
                f"{original_func.__name__}"
            )

        if input_modal is _SENTINEL:
            type_hints = typing.get_type_hints(original_func)
            if len(type_hints) >= 1:
                input_modal = type_hints[list(type_hints.keys())[0]]
            else:
                input_modal = None

        route_signature = RouteSignature(
            path=path,
            is_websocket=True,
            input_modal=input_modal,
            buffering=buffering,
            session_timeout=session_timeout,
            max_batch_size=max_batch_size,
        )
        return _fal_websocket_template(
            original_func,
            route_signature,
        )

    return marker_fn
