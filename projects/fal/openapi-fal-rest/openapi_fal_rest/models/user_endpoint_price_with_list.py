from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="UserEndpointPriceWithList")


@_attrs_define
class UserEndpointPriceWithList:
    """
    Attributes:
        endpoint (str):
        list_price (float):
        user_price (float):
        billable_unit (str):
    """

    endpoint: str
    list_price: float
    user_price: float
    billable_unit: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        endpoint = self.endpoint

        list_price = self.list_price

        user_price = self.user_price

        billable_unit = self.billable_unit

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "endpoint": endpoint,
                "list_price": list_price,
                "user_price": user_price,
                "billable_unit": billable_unit,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        endpoint = d.pop("endpoint")

        list_price = d.pop("list_price")

        user_price = d.pop("user_price")

        billable_unit = d.pop("billable_unit")

        user_endpoint_price_with_list = cls(
            endpoint=endpoint,
            list_price=list_price,
            user_price=user_price,
            billable_unit=billable_unit,
        )

        user_endpoint_price_with_list.additional_properties = d
        return user_endpoint_price_with_list

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
