from enum import Enum


class BillingStatus(str, Enum):
    BURST_PENDING = "BURST_PENDING"
    ERROR = "ERROR"
    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    SKIP = "SKIP"
    SURGE_PENDING = "SURGE_PENDING"
    WAITING = "WAITING"

    def __str__(self) -> str:
        return str(self.value)
