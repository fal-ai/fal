from enum import Enum


class UpdateVercelResourceRequestStatus(str, Enum):
    ERROR = "error"
    PENDING = "pending"
    READY = "ready"
    RESUMED = "resumed"
    SUSPENDED = "suspended"
    UNINSTALLED = "uninstalled"

    def __str__(self) -> str:
        return str(self.value)
