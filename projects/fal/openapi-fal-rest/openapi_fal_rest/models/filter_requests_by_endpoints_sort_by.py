from enum import Enum


class FilterRequestsByEndpointsSortBy(str, Enum):
    DURATION = "duration"
    ENDED_AT = "ended_at"

    def __str__(self) -> str:
        return str(self.value)
