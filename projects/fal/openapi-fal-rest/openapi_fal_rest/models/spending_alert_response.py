import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.spending_alert_period import SpendingAlertPeriod
from ..types import UNSET, Unset

T = TypeVar("T", bound="SpendingAlertResponse")


@_attrs_define
class SpendingAlertResponse:
    """
    Attributes:
        id (str):
        user_id (str):
        period (SpendingAlertPeriod):
        threshold_cents (int):
        is_enabled (bool):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        last_triggered_at (Union[Unset, datetime.datetime]):
    """

    id: str
    user_id: str
    period: SpendingAlertPeriod
    threshold_cents: int
    is_enabled: bool
    created_at: datetime.datetime
    updated_at: datetime.datetime
    last_triggered_at: Union[Unset, datetime.datetime] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        user_id = self.user_id

        period = self.period.value

        threshold_cents = self.threshold_cents

        is_enabled = self.is_enabled

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        last_triggered_at: Union[Unset, str] = UNSET
        if not isinstance(self.last_triggered_at, Unset):
            last_triggered_at = self.last_triggered_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "user_id": user_id,
                "period": period,
                "threshold_cents": threshold_cents,
                "is_enabled": is_enabled,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if last_triggered_at is not UNSET:
            field_dict["last_triggered_at"] = last_triggered_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id")

        user_id = d.pop("user_id")

        period = SpendingAlertPeriod(d.pop("period"))

        threshold_cents = d.pop("threshold_cents")

        is_enabled = d.pop("is_enabled")

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        _last_triggered_at = d.pop("last_triggered_at", UNSET)
        last_triggered_at: Union[Unset, datetime.datetime]
        if isinstance(_last_triggered_at, Unset):
            last_triggered_at = UNSET
        else:
            last_triggered_at = isoparse(_last_triggered_at)

        spending_alert_response = cls(
            id=id,
            user_id=user_id,
            period=period,
            threshold_cents=threshold_cents,
            is_enabled=is_enabled,
            created_at=created_at,
            updated_at=updated_at,
            last_triggered_at=last_triggered_at,
        )

        spending_alert_response.additional_properties = d
        return spending_alert_response

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
