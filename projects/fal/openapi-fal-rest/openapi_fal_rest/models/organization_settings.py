from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="OrganizationSettings")


@_attrs_define
class OrganizationSettings:
    """
    Attributes:
        auto_allow_enterprise_ready (bool):
        notify_enterprise_ready (bool):
    """

    auto_allow_enterprise_ready: bool
    notify_enterprise_ready: bool
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        auto_allow_enterprise_ready = self.auto_allow_enterprise_ready

        notify_enterprise_ready = self.notify_enterprise_ready

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "auto_allow_enterprise_ready": auto_allow_enterprise_ready,
                "notify_enterprise_ready": notify_enterprise_ready,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        auto_allow_enterprise_ready = d.pop("auto_allow_enterprise_ready")

        notify_enterprise_ready = d.pop("notify_enterprise_ready")

        organization_settings = cls(
            auto_allow_enterprise_ready=auto_allow_enterprise_ready,
            notify_enterprise_ready=notify_enterprise_ready,
        )

        organization_settings.additional_properties = d
        return organization_settings

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
