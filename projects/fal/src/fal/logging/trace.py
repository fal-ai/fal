from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from grpc_interceptor import ClientCallDetails, ClientInterceptor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from fal.logging import get_logger

logger = get_logger(__name__)

provider = TracerProvider()
# The line below can be used in dev to inspect opentelemetry result
# It must be imported from opentelemetry.sdk.trace.export
# processor = BatchSpanProcessor(ConsoleSpanExporter())
trace.set_tracer_provider(provider)
get_tracer = trace.get_tracer


@dataclass
class TraceContext:
    trace_id: str
    span_id: str
    invocation_id: str


def get_current_span_context() -> TraceContext | None:
    current_span = trace.get_current_span()
    if current_span is not None and current_span.is_recording():
        return TraceContext(
            trace_id=str(current_span.context.trace_id & 0xFFFFFFFFFFFFFFFF),
            span_id=str(current_span.context.span_id),
            invocation_id=current_span.attributes["invocation_id"],
        )
    return None


class TraceContextInterceptor(ClientInterceptor):
    def intercept(
        self,
        method: Callable,
        request_or_iterator: Any,
        call_details: ClientCallDetails,
    ):
        current_span = get_current_span_context()

        if current_span is not None:
            new_details = call_details._replace(
                metadata=(
                    *(call_details.metadata or []),
                    ("x-datadog-trace-id", current_span.trace_id),
                    ("x-datadog-parent-id", current_span.span_id),
                    ("x-fal-invocation-id", current_span.invocation_id),
                )
            )
            call_details = new_details

        logger.debug("Calling %s", call_details)
        return method(request_or_iterator, call_details)
