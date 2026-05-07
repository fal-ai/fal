from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="BillingSpendingAlertTriggeredPayload")


@_attrs_define
class BillingSpendingAlertTriggeredPayload:
    """
    Attributes:
        period (Union[Unset, str]):
        threshold (Union[Unset, str]):
        current_spending (Union[Unset, str]):
    """

    period: Union[Unset, str] = UNSET
    threshold: Union[Unset, str] = UNSET
    current_spending: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        period = self.period

        threshold = self.threshold

        current_spending = self.current_spending

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if period is not UNSET:
            field_dict["period"] = period
        if threshold is not UNSET:
            field_dict["threshold"] = threshold
        if current_spending is not UNSET:
            field_dict["current_spending"] = current_spending

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        period = d.pop("period", UNSET)

        threshold = d.pop("threshold", UNSET)

        current_spending = d.pop("current_spending", UNSET)

        billing_spending_alert_triggered_payload = cls(
            period=period,
            threshold=threshold,
            current_spending=current_spending,
        )

        billing_spending_alert_triggered_payload.additional_properties = d
        return billing_spending_alert_triggered_payload

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
