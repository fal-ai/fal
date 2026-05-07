from enum import Enum


class EndpointAccessControlStatus(str, Enum):
    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"
    INHERIT = "INHERIT"

    def __str__(self) -> str:
        return str(self.value)
