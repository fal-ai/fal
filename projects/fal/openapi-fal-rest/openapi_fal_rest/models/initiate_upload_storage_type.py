from enum import Enum


class InitiateUploadStorageType(str, Enum):
    FAL_CDN = "fal-cdn"
    FAL_CDN_V3 = "fal-cdn-v3"
    GCS = "gcs"

    def __str__(self) -> str:
        return str(self.value)
