from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="CustomerAddress")


@_attrs_define
class CustomerAddress:
    """
    Attributes:
        line1 (str):
        line2 (str):
        city (str):
        state (str):
        postal_code (str):
        country (str):
    """

    line1: str
    line2: str
    city: str
    state: str
    postal_code: str
    country: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        line1 = self.line1

        line2 = self.line2

        city = self.city

        state = self.state

        postal_code = self.postal_code

        country = self.country

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "line1": line1,
                "line2": line2,
                "city": city,
                "state": state,
                "postal_code": postal_code,
                "country": country,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        line1 = d.pop("line1")

        line2 = d.pop("line2")

        city = d.pop("city")

        state = d.pop("state")

        postal_code = d.pop("postal_code")

        country = d.pop("country")

        customer_address = cls(
            line1=line1,
            line2=line2,
            city=city,
            state=state,
            postal_code=postal_code,
            country=country,
        )

        customer_address.additional_properties = d
        return customer_address

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
