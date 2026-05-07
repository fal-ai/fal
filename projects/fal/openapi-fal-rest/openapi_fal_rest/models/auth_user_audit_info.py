import datetime
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

if TYPE_CHECKING:
    from ..models.auth_user_audit_info_changed import AuthUserAuditInfoChanged


T = TypeVar("T", bound="AuthUserAuditInfo")


@_attrs_define
class AuthUserAuditInfo:
    """
    Attributes:
        changed (AuthUserAuditInfoChanged):
        recorded_at (datetime.datetime):
    """

    changed: "AuthUserAuditInfoChanged"
    recorded_at: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        changed = self.changed.to_dict()

        recorded_at = self.recorded_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "changed": changed,
                "recorded_at": recorded_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.auth_user_audit_info_changed import AuthUserAuditInfoChanged

        d = src_dict.copy()
        changed = AuthUserAuditInfoChanged.from_dict(d.pop("changed"))

        recorded_at = isoparse(d.pop("recorded_at"))

        auth_user_audit_info = cls(
            changed=changed,
            recorded_at=recorded_at,
        )

        auth_user_audit_info.additional_properties = d
        return auth_user_audit_info

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
