from __future__ import annotations

from .handlers import (
    BaseExceptionHandler,
    FalServerlessExceptionHandler,
    GrpcExceptionHandler,
)


class ApplicationExceptionHandler:
    """Handle exceptions top-level exceptions.

    This exception handler is capable of handling, i.e. customize the output
    and add behavior, of any type of exception. Click handles all `ClickException`
    types by default, but prints the stack for other exception not wrapped in ClickException.

    The handler also allows for central metrics and logging collection.
    """

    _handlers: list[BaseExceptionHandler] = [
        GrpcExceptionHandler(),
        FalServerlessExceptionHandler(),
    ]

    def handle(self, exception):
        match_handler: BaseExceptionHandler = next(
            (h for h in self._handlers if h.should_handle(exception)),
            BaseExceptionHandler(),
        )
        match_handler.handle(exception)
