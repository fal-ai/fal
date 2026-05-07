from enum import Enum


class PolicyPreset(str, Enum):
    API = "API"
    BILLING = "BILLING"
    COMPUTE = "COMPUTE"
    DEPLOY = "DEPLOY"
    FULL = "FULL"
    READONLY = "READONLY"
    SERVERLESS = "SERVERLESS"

    def __str__(self) -> str:
        return str(self.value)
