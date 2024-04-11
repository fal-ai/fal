class FalTookitException(Exception):
    """Base exception for all toolkit exceptions"""

    pass


class FileUploadException(FalTookitException):
    """Raised when file upload fails"""

    pass
