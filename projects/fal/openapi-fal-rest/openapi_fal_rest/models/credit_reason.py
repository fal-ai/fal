from enum import Enum


class CreditReason(str, Enum):
    CONTRACT = "CONTRACT"
    COUPON = "COUPON"
    INTERNAL = "INTERNAL"
    MARKETING = "MARKETING"
    MAX_CREDIT = "MAX_CREDIT"
    PRE_SALES = "PRE_SALES"
    PURCHASE = "PURCHASE"
    REGISTRATION = "REGISTRATION"
    SUPPORT = "SUPPORT"
    VERCEL = "VERCEL"

    def __str__(self) -> str:
        return str(self.value)
