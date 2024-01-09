from __future__ import annotations

import inspect
import os
import fal.api
from fal.toolkit import mainify
from fastapi import FastAPI
from typing import Any, NamedTuple, Callable, TypeVar, ClassVar


EndpointT = TypeVar("EndpointT", bound=Callable[..., Any])


def wrap_app(cls: type[App], **kwargs) -> fal.api.IsolatedFunction:
    def initialize_and_serve():
        app = cls()
        app.serve()

    wrapper = fal.api.function(
        "virtualenv",
        requirements=cls.requirements,
        machine_type=cls.machine_type,
        **cls.host_kwargs,
        **kwargs,
        serve=True,
    )
    return wrapper(initialize_and_serve).on(
        serve=False,
        exposed_port=8080,
    )


@mainify
class RouteSignature(NamedTuple):
    path: str


@mainify
class App:
    requirements: ClassVar[list[str]] = []
    machine_type: ClassVar[str] = "S"
    host_kwargs: ClassVar[dict[str, Any]] = {}

    def __init_subclass__(cls, **kwargs):
        cls.host_kwargs = kwargs

    def __init__(self):
        if not os.getenv("IS_ISOLATE_AGENT"):
            raise NotImplementedError(
                "Running apps through SDK is not implemented yet."
            )

    def setup(self):
        """Setup the application before serving."""

    def serve(self) -> None:
        import uvicorn

        app = self._build_app()
        self.setup()
        uvicorn.run(app, host="0.0.0.0", port=8080)

    def _build_app(self) -> FastAPI:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        _app = FastAPI()

        _app.add_middleware(
            CORSMiddleware,
            allow_credentials=True,
            allow_headers=("*"),
            allow_methods=("*"),
            allow_origins=("*"),
        )

        routes: dict[RouteSignature, Callable[..., Any]] = {
            signature: endpoint
            for _, endpoint in inspect.getmembers(self, inspect.ismethod)
            if (signature := getattr(endpoint, "route_signature", None))
        }
        if not routes:
            raise ValueError("An application must have at least one route!")

        for signature, endpoint in routes.items():
            _app.add_api_route(
                signature.path,
                endpoint,
                name=endpoint.__name__,
                methods=["POST"],
            )

        return _app

    def openapi(self) -> dict[str, Any]:
        """
        Build the OpenAPI specification for the served function.
        Attach needed metadata for a better integration to fal.
        """
        app = self._build_app()
        spec = app.openapi()
        self._mark_order_openapi(spec)
        return spec

    def _mark_order_openapi(self, spec: dict[str, Any]):
        """
        Add x-fal-order-* keys to the OpenAPI specification to help the rendering of UI.

        NOTE: We rely on the fact that fastapi and Python dicts keep the order of properties.
        """

        def mark_order(obj: dict[str, Any], key: str):
            obj[f"x-fal-order-{key}"] = list(obj[key].keys())

        mark_order(spec, "paths")

        def order_schema_object(schema: dict[str, Any]):
            """
            Mark the order of properties in the schema object.
            They can have 'allOf', 'properties' or '$ref' key.
            """
            if "allOf" in schema:
                for sub_schema in schema["allOf"]:
                    order_schema_object(sub_schema)
            if "properties" in schema:
                mark_order(schema, "properties")

        for key in spec["components"].get("schemas") or {}:
            order_schema_object(spec["components"]["schemas"][key])

        return spec


@mainify
def endpoint(path: str) -> Callable[[EndpointT], EndpointT]:
    """Designate the decorated function as an application endpoint."""

    def marker_fn(callable: EndpointT) -> EndpointT:
        if hasattr(callable, "route_signature"):
            raise ValueError(
                f"Can't set multiple routes for the same function: {callable.__name__}"
            )

        callable.route_signature = RouteSignature(path=path)  # type: ignore
        return callable

    return marker_fn
