from enum import Enum


class RunType(str, Enum):
    CUSTOM = "custom"
    SHARED = "shared"

    def __str__(self) -> str:
        return str(self.value)
