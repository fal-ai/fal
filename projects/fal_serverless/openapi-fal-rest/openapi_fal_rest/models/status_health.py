from enum import Enum


class StatusHealth(str, Enum):
    DEGRADED = "DEGRADED"
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"

    def __str__(self) -> str:
        return str(self.value)
