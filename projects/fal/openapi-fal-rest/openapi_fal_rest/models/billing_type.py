from enum import Enum


class BillingType(str, Enum):
    INVOICE = "invoice"
    TOP_UP = "top_up"

    def __str__(self) -> str:
        return str(self.value)
