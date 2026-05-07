from enum import Enum


class ProvisionVercelResourceResponseStatus(str, Enum):
    ERROR = "error"
    PENDING = "pending"
    READY = "ready"
    RESUMED = "resumed"
    SUSPENDED = "suspended"
    UNINSTALLED = "uninstalled"

    def __str__(self) -> str:
        return str(self.value)
