class FalTookitException(Exception):
    """Base exception for all toolkit exceptions"""

    pass


class FileUploadException(FalTookitException):
    """Raised when file upload fails"""

    pass


class FileUploadFallbackException(FileUploadException):
    """Raised when the primary repository and every fallback failed; the
    message lists each attempt's error in order."""

    pass


class KVStoreException(FalTookitException):
    """Raised when KV store operation fails"""

    pass
