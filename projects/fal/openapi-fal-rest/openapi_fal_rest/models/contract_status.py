from enum import Enum


class ContractStatus(str, Enum):
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    INACTIVE = "INACTIVE"

    def __str__(self) -> str:
        return str(self.value)
