from enum import Enum


class PaymentVerificationStatus(str, Enum):
    EXEMPT = "EXEMPT"
    FAILED = "FAILED"
    NOT_STARTED = "NOT_STARTED"
    PENDING_VERIFICATION = "PENDING_VERIFICATION"
    VERIFIED = "VERIFIED"

    def __str__(self) -> str:
        return str(self.value)
