from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.application_auth_mode import ApplicationAuthMode

T = TypeVar("T", bound="AuthOverrideBody")


@_attrs_define
class AuthOverrideBody:
    """
    Attributes:
        auth_mode (ApplicationAuthMode):
    """

    auth_mode: ApplicationAuthMode
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        auth_mode = self.auth_mode.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "auth_mode": auth_mode,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        auth_mode = ApplicationAuthMode(d.pop("auth_mode"))

        auth_override_body = cls(
            auth_mode=auth_mode,
        )

        auth_override_body.additional_properties = d
        return auth_override_body

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
