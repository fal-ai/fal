from enum import Enum


class AdminListEnterpriseContractsPaginateSortBy(str, Enum):
    ENDS_AT = "ends_at"
    MONTHLY_COMMITMENT = "monthly_commitment"
    STARTS_AT = "starts_at"
    TOTAL_COMMITMENT = "total_commitment"

    def __str__(self) -> str:
        return str(self.value)
