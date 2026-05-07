from enum import Enum


class EndpointProviderType(str, Enum):
    FAL = "fal"
    PARTNER = "partner"

    def __str__(self) -> str:
        return str(self.value)
