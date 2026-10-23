from dataclasses import dataclass
from typing import Dict, Optional, Type

import grpc

from ._base import FalServerlessException


@dataclass
class VolumeException(FalServerlessException):
    """A volume request the server (or local validation) rejected.

    Attributes:
        message: Human-readable description from the server or validator.
        reason: The ``VOLUME_*`` reason code.
        http_status: HTTP status when the error came over REST.
        grpc_status: gRPC status when the error came over gRPC.
        volume_name: The volume involved, when the client knows it.
        mount_path: The mount path involved, when the client knows it.
    """

    message: str
    reason: str = "VOLUME_ERROR"
    http_status: Optional[int] = None
    grpc_status: Optional[grpc.StatusCode] = None
    volume_name: Optional[str] = None
    mount_path: Optional[str] = None

    def __str__(self) -> str:
        return self.message


# The two local-validation errors are also ValueErrors so callers that treat
# bad configuration as ValueError (e.g. pyproject parsing) keep working.
@dataclass
class VolumeInvalidNameError(VolumeException, ValueError):
    reason: str = "VOLUME_INVALID_NAME"


@dataclass
class VolumeInvalidMountPathError(VolumeException, ValueError):
    reason: str = "VOLUME_INVALID_MOUNT_PATH"


@dataclass
class VolumeNotFoundError(VolumeException):
    reason: str = "VOLUME_NOT_FOUND"


@dataclass
class VolumeUnresolvableError(VolumeException):
    reason: str = "VOLUME_UNRESOLVABLE"


@dataclass
class VolumeAlreadyExistsError(VolumeException):
    reason: str = "VOLUME_ALREADY_EXISTS"


@dataclass
class VolumeAccountNotEligibleError(VolumeException):
    reason: str = "VOLUME_ACCOUNT_NOT_ELIGIBLE"


@dataclass
class VolumeControlPlaneUnavailableError(VolumeException):
    reason: str = "VOLUME_CONTROL_PLANE_UNAVAILABLE"


VOLUME_ERRORS_BY_REASON: Dict[str, Type[VolumeException]] = {
    cls.reason: cls
    for cls in (
        VolumeInvalidNameError,
        VolumeInvalidMountPathError,
        VolumeNotFoundError,
        VolumeUnresolvableError,
        VolumeAlreadyExistsError,
        VolumeAccountNotEligibleError,
        VolumeControlPlaneUnavailableError,
    )
}


def volume_error_from_reason(
    reason: str,
    message: str,
    *,
    http_status: Optional[int] = None,
    grpc_status: Optional[grpc.StatusCode] = None,
    volume_name: Optional[str] = None,
) -> Optional[VolumeException]:
    """Return the typed error for a known reason, or None for an unknown one."""
    cls = VOLUME_ERRORS_BY_REASON.get(reason)
    if cls is None:
        return None
    return cls(
        message,
        http_status=http_status,
        grpc_status=grpc_status,
        volume_name=volume_name,
    )


def volume_error_from_grpc(
    details: str, grpc_status: grpc.StatusCode
) -> Optional[VolumeException]:
    """Parse the ``<REASON>: <message>`` form the controller puts in gRPC details."""
    reason, _, message = details.partition(":")
    return volume_error_from_reason(
        reason.strip(), message.strip() or details, grpc_status=grpc_status
    )
