from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="SpendingAlertV2Update")


@_attrs_define
class SpendingAlertV2Update:
    """
    Attributes:
        threshold_cents (Union[Unset, int]):
        is_enabled (Union[Unset, bool]):
        webhook_url (Union[Unset, str]):
        auto_lock (Union[Unset, bool]):
    """

    threshold_cents: Union[Unset, int] = UNSET
    is_enabled: Union[Unset, bool] = UNSET
    webhook_url: Union[Unset, str] = UNSET
    auto_lock: Union[Unset, bool] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        threshold_cents = self.threshold_cents

        is_enabled = self.is_enabled

        webhook_url = self.webhook_url

        auto_lock = self.auto_lock

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if threshold_cents is not UNSET:
            field_dict["threshold_cents"] = threshold_cents
        if is_enabled is not UNSET:
            field_dict["is_enabled"] = is_enabled
        if webhook_url is not UNSET:
            field_dict["webhook_url"] = webhook_url
        if auto_lock is not UNSET:
            field_dict["auto_lock"] = auto_lock

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        threshold_cents = d.pop("threshold_cents", UNSET)

        is_enabled = d.pop("is_enabled", UNSET)

        webhook_url = d.pop("webhook_url", UNSET)

        auto_lock = d.pop("auto_lock", UNSET)

        spending_alert_v2_update = cls(
            threshold_cents=threshold_cents,
            is_enabled=is_enabled,
            webhook_url=webhook_url,
            auto_lock=auto_lock,
        )

        spending_alert_v2_update.additional_properties = d
        return spending_alert_v2_update

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
