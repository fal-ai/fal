from enum import Enum


class DeploymentStrategy(str, Enum):
    RECREATE = "recreate"
    ROLLING = "rolling"

    def __str__(self) -> str:
        return str(self.value)
