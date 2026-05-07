from enum import Enum


class Region(str, Enum):
    EU_NORTH = "eu-north"
    EU_WEST = "eu-west"
    OTHER = "other"
    US_CENTRAL = "us-central"
    US_EAST = "us-east"
    US_WEST = "us-west"

    def __str__(self) -> str:
        return str(self.value)
