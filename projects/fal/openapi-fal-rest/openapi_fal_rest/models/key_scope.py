from enum import Enum


class KeyScope(str, Enum):
    ADMIN = "ADMIN"
    API = "API"

    def __str__(self) -> str:
        return str(self.value)
