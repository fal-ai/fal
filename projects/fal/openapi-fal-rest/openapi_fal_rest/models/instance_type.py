from enum import Enum


class InstanceType(str, Enum):
    CONTAINER = "container"
    GPU_1X_H100_SXM5 = "gpu_1x_h100_sxm5"
    GPU_8X_H100_SXM5 = "gpu_8x_h100_sxm5"

    def __str__(self) -> str:
        return str(self.value)
