from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.application_auth_mode import ApplicationAuthMode

T = TypeVar("T", bound="UserAppInfoUserAuthMode")


@_attrs_define
class UserAppInfoUserAuthMode:
    """ """

    additional_properties: dict[str, ApplicationAuthMode] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.value

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_app_info_user_auth_mode = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = ApplicationAuthMode(prop_dict)

            additional_properties[prop_name] = additional_property

        user_app_info_user_auth_mode.additional_properties = additional_properties
        return user_app_info_user_auth_mode

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> ApplicationAuthMode:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: ApplicationAuthMode) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
