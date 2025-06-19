class FalTookitException(Exception):
    """Base exception for all toolkit exceptions"""

    pass


class FileUploadException(FalTookitException):
    """Raised when file upload fails"""

    pass


class KVStoreException(FalTookitException):
    """Raised when KV store operation fails"""

    pass
