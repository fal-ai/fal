from enum import Enum


class AggregateRequestsByAppMetric(str, Enum):
    ERRORS = "errors"
    REQUESTS = "requests"

    def __str__(self) -> str:
        return str(self.value)
