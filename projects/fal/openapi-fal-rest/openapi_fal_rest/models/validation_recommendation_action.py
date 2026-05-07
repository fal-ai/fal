from enum import Enum


class ValidationRecommendationAction(str, Enum):
    ADD_ADMIN = "add-admin"
    ADD_TEAM = "add-team"
    REMOVE_TEAM = "remove-team"
    SYNC_ACCOUNT_TYPE = "sync-account-type"
    SYNC_MODEL_ACCESS_CONTROLS_ENABLED = "sync-model-access-controls-enabled"

    def __str__(self) -> str:
        return str(self.value)
