from enum import Enum


class Sector(str, Enum):
    SECTOR_1 = "sector_1"
    SECTOR_2 = "sector_2"
    SECTOR_3 = "sector_3"
    SECTOR_4 = "sector_4"

    def __str__(self) -> str:
        return str(self.value)
