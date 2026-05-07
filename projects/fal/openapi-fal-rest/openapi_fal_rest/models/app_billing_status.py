from enum import Enum


class AppBillingStatus(str, Enum):
    CHARGE = "charge"
    SKIP = "skip"

    def __str__(self) -> str:
        return str(self.value)
