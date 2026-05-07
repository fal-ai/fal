from enum import Enum


class InstanceStatus(str, Enum):
    INIT = "init"
    PENDING = "pending"
    PROVISIONING = "provisioning"
    READY = "ready"
    STOPPED = "stopped"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return str(self.value)
