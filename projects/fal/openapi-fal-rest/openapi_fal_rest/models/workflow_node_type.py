from enum import Enum


class WorkflowNodeType(str, Enum):
    DISPLAY = "display"
    RUN = "run"

    def __str__(self) -> str:
        return str(self.value)
