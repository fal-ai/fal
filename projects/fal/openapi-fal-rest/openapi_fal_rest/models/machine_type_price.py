from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.endpoint_provider_type import EndpointProviderType

T = TypeVar("T", bound="MachineTypePrice")


@_attrs_define
class MachineTypePrice:
    """
    Attributes:
        machine_type (str):
        price (float):
        billable_unit (str):
        provider (EndpointProviderType):
    """

    machine_type: str
    price: float
    billable_unit: str
    provider: EndpointProviderType
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        machine_type = self.machine_type

        price = self.price

        billable_unit = self.billable_unit

        provider = self.provider.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "machine_type": machine_type,
                "price": price,
                "billable_unit": billable_unit,
                "provider": provider,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        machine_type = d.pop("machine_type")

        price = d.pop("price")

        billable_unit = d.pop("billable_unit")

        provider = EndpointProviderType(d.pop("provider"))

        machine_type_price = cls(
            machine_type=machine_type,
            price=price,
            billable_unit=billable_unit,
            provider=provider,
        )

        machine_type_price.additional_properties = d
        return machine_type_price

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
