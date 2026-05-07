from enum import Enum


class ResolvedEndpointAccessControlStatus(str, Enum):
    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"

    def __str__(self) -> str:
        return str(self.value)
