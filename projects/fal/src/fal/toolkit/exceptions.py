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
        super().__init__(message)
        self.failures = failures


class KVStoreException(FalTookitException):
    """Raised when KV store operation fails"""

    pass
