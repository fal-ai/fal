from enum import Enum


class LockReason(str, Enum):
    ADMIN_LOCK_PLEASE_CONTACT_HELLOFAL_AI = "Admin lock. Please contact hello@fal.ai."
    EXHAUSTED_BALANCE_TOP_UP_YOUR_BALANCE_AT_FAL_AIDASHBOARDBILLING = (
        "Exhausted balance. Top up your balance at fal.ai/dashboard/billing"
    )
    PLEASE_ADD_A_PAYMENT_METHOD_AT_FAL_AIDASHBOARDBILLING = "Please add a payment method at fal.ai/dashboard/billing"
    THIS_ACCOUNT_HAS_BEEN_ARCHIVED_PLEASE_CONTACT_HELLOFAL_AI = (
        "This account has been archived. Please contact hello@fal.ai."
    )
    UNKNOWN_PLEASE_CONTACT_HELLOFAL_AI = "Unknown. Please contact hello@fal.ai."
    USER_BUDGET_IS_EXCEEDED_ADJUST_IT_AT_FAL_AIDASHBOARDBILLING = (
        "User budget is exceeded. Adjust it at fal.ai/dashboard/billing"
    )

    def __str__(self) -> str:
        return str(self.value)
