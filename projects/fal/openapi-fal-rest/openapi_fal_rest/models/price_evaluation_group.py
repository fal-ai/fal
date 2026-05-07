from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="PriceEvaluationGroup")


@_attrs_define
class PriceEvaluationGroup:
    """
    Attributes:
        price_name (str):
        amount (str):
        grouping_key (str):
        quantity (float):
    """

    price_name: str
    amount: str
    grouping_key: str
    quantity: float
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        price_name = self.price_name

        amount = self.amount

        grouping_key = self.grouping_key

        quantity = self.quantity

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "price_name": price_name,
                "amount": amount,
                "grouping_key": grouping_key,
                "quantity": quantity,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        price_name = d.pop("price_name")

        amount = d.pop("amount")

        grouping_key = d.pop("grouping_key")

        quantity = d.pop("quantity")

        price_evaluation_group = cls(
            price_name=price_name,
            amount=amount,
            grouping_key=grouping_key,
            quantity=quantity,
        )

        price_evaluation_group.additional_properties = d
        return price_evaluation_group

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
