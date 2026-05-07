from enum import Enum


class ListRegistryEndpointsPaginateSort(str, Enum):
    RECENT = "recent"
    RELEVANT = "relevant"

    def __str__(self) -> str:
        return str(self.value)
