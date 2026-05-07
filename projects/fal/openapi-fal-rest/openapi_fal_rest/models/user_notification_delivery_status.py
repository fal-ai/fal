from enum import Enum


class UserNotificationDeliveryStatus(str, Enum):
    DELIVERED = "delivered"
    FAILED = "failed"
    READ = "read"
    SENT = "sent"

    def __str__(self) -> str:
        return str(self.value)
