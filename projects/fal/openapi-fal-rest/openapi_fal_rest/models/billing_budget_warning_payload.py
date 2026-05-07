from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="BillingBudgetWarningPayload")


@_attrs_define
class BillingBudgetWarningPayload:
    """
    Attributes:
        balance_threshold (Union[Unset, float]):
    """

    balance_threshold: Union[Unset, float] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        balance_threshold = self.balance_threshold

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if balance_threshold is not UNSET:
            field_dict["balance_threshold"] = balance_threshold

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        balance_threshold = d.pop("balance_threshold", UNSET)

        billing_budget_warning_payload = cls(
            balance_threshold=balance_threshold,
        )

        billing_budget_warning_payload.additional_properties = d
        return billing_budget_warning_payload

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
