from enum import Enum


class DiscountType(str, Enum):
    PLATFORM = "PLATFORM"
    PLATFORM_ALL = "PLATFORM_ALL"
    REBATE_AMOUNT = "REBATE_AMOUNT"
    REBATE_PCT = "REBATE_PCT"

    def __str__(self) -> str:
        return str(self.value)
