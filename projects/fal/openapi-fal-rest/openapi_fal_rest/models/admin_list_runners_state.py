from enum import Enum


class AdminListRunnersState(str, Enum):
    DOCKER_PULL = "DOCKER_PULL"
    DRAINING = "DRAINING"
    FAILED = "FAILED"
    FAILURE_DELAY = "FAILURE_DELAY"
    IDLE = "IDLE"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SETUP = "SETUP"
    TERMINATED = "TERMINATED"
    TERMINATING = "TERMINATING"

    def __str__(self) -> str:
        return str(self.value)
