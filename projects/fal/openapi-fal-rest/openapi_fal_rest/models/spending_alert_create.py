from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.spending_alert_period import SpendingAlertPeriod

T = TypeVar("T", bound="SpendingAlertCreate")


@_attrs_define
class SpendingAlertCreate:
    """
    Attributes:
        period (SpendingAlertPeriod):
        threshold_cents (int):
    """

    period: SpendingAlertPeriod
    threshold_cents: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        period = self.period.value

        threshold_cents = self.threshold_cents

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "period": period,
                "threshold_cents": threshold_cents,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        period = SpendingAlertPeriod(d.pop("period"))

        threshold_cents = d.pop("threshold_cents")

        spending_alert_create = cls(
            period=period,
            threshold_cents=threshold_cents,
        )

        spending_alert_create.additional_properties = d
        return spending_alert_create

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
