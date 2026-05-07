from enum import Enum


class EndpointAccessControlContext(str, Enum):
    API = "api"
    UI = "ui"

    def __str__(self) -> str:
        return str(self.value)
