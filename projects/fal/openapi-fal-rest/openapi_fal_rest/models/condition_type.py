from enum import Enum


class ConditionType(str, Enum):
    CONTAINS = "CONTAINS"
    EXACT = "EXACT"

    def __str__(self) -> str:
        return str(self.value)
