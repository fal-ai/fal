from enum import Enum


class SpendingSubjectType(str, Enum):
    API_KEY = "api_key"
    AUTH_USER = "auth_user"
    TEAM = "team"

    def __str__(self) -> str:
        return str(self.value)
