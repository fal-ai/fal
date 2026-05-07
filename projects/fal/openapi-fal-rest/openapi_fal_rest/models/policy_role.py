from enum import Enum


class PolicyRole(str, Enum):
    ADMIN = "Admin"
    BILLING = "Billing"
    CREATOR = "Creator"
    DEVELOPER = "Developer"
    VIEWER = "Viewer"

    def __str__(self) -> str:
        return str(self.value)
