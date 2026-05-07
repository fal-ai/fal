from enum import Enum


class CDNObjectACLDecision(str, Enum):
    ALLOW = "allow"
    FORBID = "forbid"
    HIDE = "hide"

    def __str__(self) -> str:
        return str(self.value)
