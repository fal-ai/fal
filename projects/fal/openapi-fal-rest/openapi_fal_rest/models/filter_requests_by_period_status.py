from enum import Enum


class FilterRequestsByPeriodStatus(str, Enum):
    ERROR = "error"
    SUCCESS = "success"
    USER_ERROR = "user_error"

    def __str__(self) -> str:
        return str(self.value)
