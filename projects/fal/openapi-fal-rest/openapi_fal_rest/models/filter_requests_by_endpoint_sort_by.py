from enum import Enum


class FilterRequestsByEndpointSortBy(str, Enum):
    DURATION = "duration"
    ENDED_AT = "ended_at"

    def __str__(self) -> str:
        return str(self.value)
