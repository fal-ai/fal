from enum import Enum


class AdminRole(str, Enum):
    ADMINACCESS_APPROVER = "admin:access-approver"
    ADMINCPQ = "admin:cpq"
    ADMINDEVELOPERS = "admin:developers"
    ADMINFULL = "admin:full"
    ADMINHIGH_CREDIT_GRANTER = "admin:high-credit-granter"
    ADMINLIMITED = "admin:limited"
    ADMINOPERATIONS = "admin:operations"
    ADMINORGANIZATIONS = "admin:organizations"
    ADMINREQUEST_VIEWER = "admin:request-viewer"

    def __str__(self) -> str:
        return str(self.value)
