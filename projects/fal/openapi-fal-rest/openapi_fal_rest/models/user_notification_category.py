from enum import Enum


class UserNotificationCategory(str, Enum):
    BILLING = "billing"
    COLLABORATION = "collaboration"
    MARKETING = "marketing"
    PLATFORM = "platform"
    SECURITY = "security"

    def __str__(self) -> str:
        return str(self.value)
