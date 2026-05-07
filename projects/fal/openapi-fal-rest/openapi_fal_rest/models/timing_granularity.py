from enum import Enum


class TimingGranularity(str, Enum):
    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"
    MINUTE_10 = "minute_10"
    MINUTE_5 = "minute_5"
    MONTH = "month"
    SECOND_30 = "second_30"
    WEEK = "week"

    def __str__(self) -> str:
        return str(self.value)
