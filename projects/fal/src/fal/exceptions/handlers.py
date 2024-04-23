from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from grpc import Call as RpcCall

from fal.console import console
from fal.console.icons import CROSS_ICON

if TYPE_CHECKING:
    from fal.api import UserFunctionException


ExceptionType = TypeVar("ExceptionType", bound=BaseException)


class BaseExceptionHandler(Generic[ExceptionType]):
    """Base handler defaults to the string representation of the error"""

    def should_handle(self, _: Exception) -> bool:
        return True

    def handle(self, exception: ExceptionType):
        msg = f"{CROSS_ICON} {str(exception)}"
        cause = exception.__cause__
        if cause is not None:
            msg += f": {str(cause)}"
        console.print(msg)


class GrpcExceptionHandler(BaseExceptionHandler[RpcCall]):
    """Handle GRPC errors. The user message is part of the `details()`"""

    def should_handle(self, exception: Exception) -> bool:
        return isinstance(exception, RpcCall)

    def handle(self, exception: RpcCall):
        console.print(exception.details())


class UserFunctionExceptionHandler(BaseExceptionHandler["UserFunctionException"]):
    def should_handle(self, exception: Exception) -> bool:
        from fal.api import UserFunctionException

        return isinstance(exception, UserFunctionException)

    def handle(self, exception: UserFunctionException):
        import rich

        cause = exception.__cause__
        exc = cause or exception
        tb = rich.traceback.Traceback.from_exception(
            type(exc),
            exc,
            exc.__traceback__,
        )
        console.print(tb)
        super().handle(exception)
