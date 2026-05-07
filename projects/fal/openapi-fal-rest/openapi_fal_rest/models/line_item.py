from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="LineItem")


@_attrs_define
class LineItem:
    """A single line item in an invoice section.

    Attributes:
        endpoint (str):
        quantity (float):
        amount (str):
        platform_discount (bool):
        platform_discount_percentage (Union[Unset, float]):
        platform_discount_amount (Union[Unset, float]):
        endpoint_discount (Union[Unset, bool]):  Default: False.
        endpoint_discount_percentage (Union[Unset, float]):
        endpoint_discount_amount (Union[Unset, float]):
        caller_user_id (Union[Unset, str]):
    """

    endpoint: str
    quantity: float
    amount: str
    platform_discount: bool
    platform_discount_percentage: Union[Unset, float] = UNSET
    platform_discount_amount: Union[Unset, float] = UNSET
    endpoint_discount: Union[Unset, bool] = False
    endpoint_discount_percentage: Union[Unset, float] = UNSET
    endpoint_discount_amount: Union[Unset, float] = UNSET
    caller_user_id: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        endpoint = self.endpoint

        quantity = self.quantity

        amount = self.amount

        platform_discount = self.platform_discount

        platform_discount_percentage = self.platform_discount_percentage

        platform_discount_amount = self.platform_discount_amount

        endpoint_discount = self.endpoint_discount

        endpoint_discount_percentage = self.endpoint_discount_percentage

        endpoint_discount_amount = self.endpoint_discount_amount

        caller_user_id = self.caller_user_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "endpoint": endpoint,
                "quantity": quantity,
                "amount": amount,
                "platform_discount": platform_discount,
            }
        )
        if platform_discount_percentage is not UNSET:
            field_dict["platform_discount_percentage"] = platform_discount_percentage
        if platform_discount_amount is not UNSET:
            field_dict["platform_discount_amount"] = platform_discount_amount
        if endpoint_discount is not UNSET:
            field_dict["endpoint_discount"] = endpoint_discount
        if endpoint_discount_percentage is not UNSET:
            field_dict["endpoint_discount_percentage"] = endpoint_discount_percentage
        if endpoint_discount_amount is not UNSET:
            field_dict["endpoint_discount_amount"] = endpoint_discount_amount
        if caller_user_id is not UNSET:
            field_dict["caller_user_id"] = caller_user_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        endpoint = d.pop("endpoint")

        quantity = d.pop("quantity")

        amount = d.pop("amount")

        platform_discount = d.pop("platform_discount")

        platform_discount_percentage = d.pop("platform_discount_percentage", UNSET)

        platform_discount_amount = d.pop("platform_discount_amount", UNSET)

        endpoint_discount = d.pop("endpoint_discount", UNSET)

        endpoint_discount_percentage = d.pop("endpoint_discount_percentage", UNSET)

        endpoint_discount_amount = d.pop("endpoint_discount_amount", UNSET)

        caller_user_id = d.pop("caller_user_id", UNSET)

        line_item = cls(
            endpoint=endpoint,
            quantity=quantity,
            amount=amount,
            platform_discount=platform_discount,
            platform_discount_percentage=platform_discount_percentage,
            platform_discount_amount=platform_discount_amount,
            endpoint_discount=endpoint_discount,
            endpoint_discount_percentage=endpoint_discount_percentage,
            endpoint_discount_amount=endpoint_discount_amount,
            caller_user_id=caller_user_id,
        )

        line_item.additional_properties = d
        return line_item

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
