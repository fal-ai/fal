from enum import Enum


class TeamRole(str, Enum):
    ADMIN = "Admin"
    BILLING = "Billing"
    DEVELOPER = "Developer"

    def __str__(self) -> str:
        return str(self.value)
