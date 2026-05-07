from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.application_auth_mode import ApplicationAuthMode

T = TypeVar("T", bound="AppInfoPerUserAuth")


@_attrs_define
class AppInfoPerUserAuth:
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
        app_info_per_user_auth = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = ApplicationAuthMode(prop_dict)

            additional_properties[prop_name] = additional_property

        app_info_per_user_auth.additional_properties = additional_properties
        return app_info_per_user_auth

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
