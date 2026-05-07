from enum import Enum


class UserNotificationChannel(str, Enum):
    EMAIL = "email"
    SLACK = "slack"
    TEAM_ADMIN_EMAIL = "team_admin_email"
    UI = "ui"

    def __str__(self) -> str:
        return str(self.value)
