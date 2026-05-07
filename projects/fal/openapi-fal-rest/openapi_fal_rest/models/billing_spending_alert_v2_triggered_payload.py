from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="BillingSpendingAlertV2TriggeredPayload")


@_attrs_define
class BillingSpendingAlertV2TriggeredPayload:
    """
    Attributes:
        account_nickname (Union[Unset, str]):
        period (Union[Unset, str]):
        subject_label (Union[Unset, str]):
        threshold (Union[Unset, str]):
        current_spending (Union[Unset, str]):
        auto_locked (Union[Unset, bool]):
    """

    account_nickname: Union[Unset, str] = UNSET
    period: Union[Unset, str] = UNSET
    subject_label: Union[Unset, str] = UNSET
    threshold: Union[Unset, str] = UNSET
    current_spending: Union[Unset, str] = UNSET
    auto_locked: Union[Unset, bool] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        account_nickname = self.account_nickname

        period = self.period

        subject_label = self.subject_label

        threshold = self.threshold

        current_spending = self.current_spending

        auto_locked = self.auto_locked

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if account_nickname is not UNSET:
            field_dict["account_nickname"] = account_nickname
        if period is not UNSET:
            field_dict["period"] = period
        if subject_label is not UNSET:
            field_dict["subject_label"] = subject_label
        if threshold is not UNSET:
            field_dict["threshold"] = threshold
        if current_spending is not UNSET:
            field_dict["current_spending"] = current_spending
        if auto_locked is not UNSET:
            field_dict["auto_locked"] = auto_locked

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        account_nickname = d.pop("account_nickname", UNSET)

        period = d.pop("period", UNSET)

        subject_label = d.pop("subject_label", UNSET)

        threshold = d.pop("threshold", UNSET)

        current_spending = d.pop("current_spending", UNSET)

        auto_locked = d.pop("auto_locked", UNSET)

        billing_spending_alert_v2_triggered_payload = cls(
            account_nickname=account_nickname,
            period=period,
            subject_label=subject_label,
            threshold=threshold,
            current_spending=current_spending,
            auto_locked=auto_locked,
        )

        billing_spending_alert_v2_triggered_payload.additional_properties = d
        return billing_spending_alert_v2_triggered_payload

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
