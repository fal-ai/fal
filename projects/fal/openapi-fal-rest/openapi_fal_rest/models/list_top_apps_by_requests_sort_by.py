from enum import Enum


class ListTopAppsByRequestsSortBy(str, Enum):
    CONCURRENT = "concurrent"
    REQUESTS = "requests"
    REQUEST_RATE = "request_rate"

    def __str__(self) -> str:
        return str(self.value)
