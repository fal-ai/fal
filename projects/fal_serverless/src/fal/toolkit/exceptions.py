from fal.toolkit.mainify import mainify


@mainify
class FalTookitException(Exception):
    """Base exception for all toolkit exceptions"""

    pass


@mainify
class FileUploadException(FalTookitException):
    """Raised when file upload fails"""

    pass
