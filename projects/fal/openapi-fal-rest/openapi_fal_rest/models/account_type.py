from enum import Enum


class AccountType(str, Enum):
    ENTERPRISE = "enterprise"
    ENTERPRISE_MSA = "enterprise_msa"
    SELF_SERVICE = "self_service"

    def __str__(self) -> str:
        return str(self.value)
