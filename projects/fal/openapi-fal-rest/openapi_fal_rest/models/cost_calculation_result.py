from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="CostCalculationResult")


@_attrs_define
class CostCalculationResult:
    """
    Attributes:
        billable_duration (float):
        amount (float):
        is_estimate (bool):
        price (float):
        dimension_key (str):
        dimension_value (Union[Unset, str]):
    """

    billable_duration: float
    amount: float
    is_estimate: bool
    price: float
    dimension_key: str
    dimension_value: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        billable_duration = self.billable_duration

        amount = self.amount

        is_estimate = self.is_estimate

        price = self.price

        dimension_key = self.dimension_key

        dimension_value = self.dimension_value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "billable_duration": billable_duration,
                "amount": amount,
                "is_estimate": is_estimate,
                "price": price,
                "dimension_key": dimension_key,
            }
        )
        if dimension_value is not UNSET:
            field_dict["dimension_value"] = dimension_value

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        billable_duration = d.pop("billable_duration")

        amount = d.pop("amount")

        is_estimate = d.pop("is_estimate")

        price = d.pop("price")

        dimension_key = d.pop("dimension_key")

        dimension_value = d.pop("dimension_value", UNSET)

        cost_calculation_result = cls(
            billable_duration=billable_duration,
            amount=amount,
            is_estimate=is_estimate,
            price=price,
            dimension_key=dimension_key,
            dimension_value=dimension_value,
        )

        cost_calculation_result.additional_properties = d
        return cost_calculation_result

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
