from enum import Enum


class AdminAppRequestLatencyRequestType(str, Enum):
    HTTP = "http"
    INFERENCE = "inference"

    def __str__(self) -> str:
        return str(self.value)
