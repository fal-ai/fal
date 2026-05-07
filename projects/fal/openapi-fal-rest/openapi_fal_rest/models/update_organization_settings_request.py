from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="UpdateOrganizationSettingsRequest")


@_attrs_define
class UpdateOrganizationSettingsRequest:
    """
    Attributes:
        auto_allow_enterprise_ready (Union[Unset, bool]):
        notify_enterprise_ready (Union[Unset, bool]):
    """

    auto_allow_enterprise_ready: Union[Unset, bool] = UNSET
    notify_enterprise_ready: Union[Unset, bool] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        auto_allow_enterprise_ready = self.auto_allow_enterprise_ready

        notify_enterprise_ready = self.notify_enterprise_ready

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if auto_allow_enterprise_ready is not UNSET:
            field_dict["auto_allow_enterprise_ready"] = auto_allow_enterprise_ready
        if notify_enterprise_ready is not UNSET:
            field_dict["notify_enterprise_ready"] = notify_enterprise_ready

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        auto_allow_enterprise_ready = d.pop("auto_allow_enterprise_ready", UNSET)

        notify_enterprise_ready = d.pop("notify_enterprise_ready", UNSET)

        update_organization_settings_request = cls(
            auto_allow_enterprise_ready=auto_allow_enterprise_ready,
            notify_enterprise_ready=notify_enterprise_ready,
        )

        update_organization_settings_request.additional_properties = d
        return update_organization_settings_request

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
