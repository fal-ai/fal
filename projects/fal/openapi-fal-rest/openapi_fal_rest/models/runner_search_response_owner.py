from enum import Enum


class RunnerSearchResponseOwner(str, Enum):
    CRON = "cron"
    GATEWAY = "gateway"
    GRPC_REGISTER = "grpc-register"
    GRPC_RUN = "grpc-run"

    def __str__(self) -> str:
        return str(self.value)
