from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    ERROR = "ERROR"
    INFO = "INFO"
    STDERR = "STDERR"
    STDOUT = "STDOUT"
    TRACE = "TRACE"
    WARNING = "WARNING"

    def __str__(self) -> str:
        return str(self.value)
