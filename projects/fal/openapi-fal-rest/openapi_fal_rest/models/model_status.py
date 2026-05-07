from enum import Enum


class ModelStatus(str, Enum):
    DEGRADED = "degraded"
    MAJOR_OUTAGE = "major_outage"
    OPERATIONAL = "operational"
    PARTIAL_OUTAGE = "partial_outage"

    def __str__(self) -> str:
        return str(self.value)
