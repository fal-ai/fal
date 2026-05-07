from enum import Enum


class StatsTimeframe(str, Enum):
    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"
    MONTH = "month"
    WEEK = "week"

    def __str__(self) -> str:
        return str(self.value)
