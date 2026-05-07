from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="CreditBalanceAlert")


@_attrs_define
class CreditBalanceAlert:
    """
    Attributes:
        threshold_cents (int):
        alert_id (Union[Unset, str]):
        enabled (Union[Unset, bool]):  Default: True.
    """

    threshold_cents: int
    alert_id: Union[Unset, str] = UNSET
    enabled: Union[Unset, bool] = True
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        threshold_cents = self.threshold_cents

        alert_id = self.alert_id

        enabled = self.enabled

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "threshold_cents": threshold_cents,
            }
        )
        if alert_id is not UNSET:
            field_dict["alert_id"] = alert_id
        if enabled is not UNSET:
            field_dict["enabled"] = enabled

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        threshold_cents = d.pop("threshold_cents")

        alert_id = d.pop("alert_id", UNSET)

        enabled = d.pop("enabled", UNSET)

        credit_balance_alert = cls(
            threshold_cents=threshold_cents,
            alert_id=alert_id,
            enabled=enabled,
        )

        credit_balance_alert.additional_properties = d
        return credit_balance_alert

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
