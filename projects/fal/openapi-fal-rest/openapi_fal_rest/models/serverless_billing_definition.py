from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ServerlessBillingDefinition")


@_attrs_define
class ServerlessBillingDefinition:
    """
    Attributes:
        machine_type (str):
        unit_price (float):
        is_visible (Union[Unset, bool]):  Default: False.
    """

    machine_type: str
    unit_price: float
    is_visible: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        machine_type = self.machine_type

        unit_price = self.unit_price

        is_visible = self.is_visible

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "machine_type": machine_type,
                "unit_price": unit_price,
            }
        )
        if is_visible is not UNSET:
            field_dict["is_visible"] = is_visible

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        machine_type = d.pop("machine_type")

        unit_price = d.pop("unit_price")

        is_visible = d.pop("is_visible", UNSET)

        serverless_billing_definition = cls(
            machine_type=machine_type,
            unit_price=unit_price,
            is_visible=is_visible,
        )

        serverless_billing_definition.additional_properties = d
        return serverless_billing_definition

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
