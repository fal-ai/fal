from enum import Enum


class EnrichedUserInfoLookupMethod(str, Enum):
    ALTERNATIVE_AUTH_ID = "alternative_auth_id"
    ALTERNATIVE_WORKOS_SUB = "alternative_workos_sub"
    AUTH_ID = "auth_id"
    UNKNOWN = "unknown"
    WORKOS_SUB = "workos_sub"

    def __str__(self) -> str:
        return str(self.value)
