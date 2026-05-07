from enum import Enum


class DiscountBucketType(str, Enum):
    COMPUTE = "compute"
    ENDPOINT = "endpoint"
    SERVERLESS = "serverless"

    def __str__(self) -> str:
        return str(self.value)
