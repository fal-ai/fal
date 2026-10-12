class FalTookitException(Exception):
    """Base exception for all toolkit exceptions"""

    pass


class FileUploadException(FalTookitException):
    """Raised when file upload fails"""

    pass


class FileUploadFallbackException(FileUploadException):
    """Raised when the primary repository and every fallback failed.

    ``failures`` holds one ``(repository label, exception)`` pair per attempt, in
    the order they were tried.
    """

    def __init__(self, message: str, failures: "list[tuple[str, Exception]]"):
        # Keep ``failures`` in ``args`` so the exception survives pickling: the
        # isolate agent serializes raised exceptions back to the client, and
        # ``BaseException.__reduce__`` re-invokes ``__init__(*args)``.
        super().__init__(message, failures)
        self.failures = failures

    def __str__(self) -> str:
        return str(self.args[0])


class KVStoreException(FalTookitException):
    """Raised when KV store operation fails"""

    pass
