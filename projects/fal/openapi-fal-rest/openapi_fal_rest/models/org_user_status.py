from enum import Enum


class OrgUserStatus(str, Enum):
    ACTIVE = "active"
    INVITE_PENDING = "invite_pending"
    UNASSIGNED = "unassigned"

    def __str__(self) -> str:
        return str(self.value)
