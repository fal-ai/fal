from __future__ import annotations

from typing import Generic, TypeVar

from grpc import Call as RpcCall
from rich.markdown import Markdown

from fal.console import console
from fal.console.icons import CROSS_ICON

from ._base import FalServerlessException

ExceptionType = TypeVar("ExceptionType")


class BaseExceptionHandler(Generic[ExceptionType]):
    """Base handler defaults to the string representation of the error"""

    def should_handle(self, _: Exception) -> bool:
        return True

    def handle(self, exception: ExceptionType):
        console.print(str(exception))


class FalServerlessExceptionHandler(BaseExceptionHandler[FalServerlessException]):
    """Handle fal Serverless exceptions"""

    def should_handle(self, exception: Exception) -> bool:
        return isinstance(exception, FalServerlessException)

    def handle(self, exception: FalServerlessException):
        console.print(f"{CROSS_ICON} {exception.message}")
        if exception.hint is not None:
            console.print(Markdown(f"**Hint:** {exception.hint}"))
            console.print()


class GrpcExceptionHandler(BaseExceptionHandler[RpcCall]):
    """Handle GRPC errors. The user message is part of the `details()`"""

    def should_handle(self, exception: Exception) -> bool:
        return isinstance(exception, RpcCall)

    def handle(self, exception: RpcCall):
        console.print(exception.details())
