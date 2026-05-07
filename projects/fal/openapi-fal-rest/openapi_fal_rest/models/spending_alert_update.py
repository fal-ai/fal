from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="SpendingAlertUpdate")


@_attrs_define
class SpendingAlertUpdate:
    """
    Attributes:
        threshold_cents (Union[Unset, int]):
        is_enabled (Union[Unset, bool]):
    """

    threshold_cents: Union[Unset, int] = UNSET
    is_enabled: Union[Unset, bool] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        threshold_cents = self.threshold_cents

        is_enabled = self.is_enabled

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if threshold_cents is not UNSET:
            field_dict["threshold_cents"] = threshold_cents
        if is_enabled is not UNSET:
            field_dict["is_enabled"] = is_enabled

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        threshold_cents = d.pop("threshold_cents", UNSET)

        is_enabled = d.pop("is_enabled", UNSET)

        spending_alert_update = cls(
            threshold_cents=threshold_cents,
            is_enabled=is_enabled,
        )

        spending_alert_update.additional_properties = d
        return spending_alert_update

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
