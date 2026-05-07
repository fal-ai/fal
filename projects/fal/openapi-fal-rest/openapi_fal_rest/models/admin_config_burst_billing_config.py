from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.admin_config_burst_billing_config_additional_property import (
        AdminConfigBurstBillingConfigAdditionalProperty,
    )


T = TypeVar("T", bound="AdminConfigBurstBillingConfig")


@_attrs_define
class AdminConfigBurstBillingConfig:
    """ """

    additional_properties: dict[str, "AdminConfigBurstBillingConfigAdditionalProperty"] = _attrs_field(
        init=False, factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.admin_config_burst_billing_config_additional_property import (
            AdminConfigBurstBillingConfigAdditionalProperty,
        )

        d = src_dict.copy()
        admin_config_burst_billing_config = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = AdminConfigBurstBillingConfigAdditionalProperty.from_dict(prop_dict)

            additional_properties[prop_name] = additional_property

        admin_config_burst_billing_config.additional_properties = additional_properties
        return admin_config_burst_billing_config

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> "AdminConfigBurstBillingConfigAdditionalProperty":
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: "AdminConfigBurstBillingConfigAdditionalProperty") -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
