from enum import Enum


class AuthTokenStorageType(str, Enum):
    FAL_CDN = "fal-cdn"
    FAL_CDN_V3 = "fal-cdn-v3"

    def __str__(self) -> str:
        return str(self.value)
