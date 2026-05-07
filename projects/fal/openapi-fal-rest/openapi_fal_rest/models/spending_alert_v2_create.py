from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.spending_alert_period import SpendingAlertPeriod
from ..models.spending_subject_type import SpendingSubjectType
from ..types import UNSET, Unset

T = TypeVar("T", bound="SpendingAlertV2Create")


@_attrs_define
class SpendingAlertV2Create:
    """
    Attributes:
        subject_type (SpendingSubjectType):
        subject_id (str):
        period (SpendingAlertPeriod):
        threshold_cents (int):
        webhook_url (Union[Unset, str]):
        auto_lock (Union[Unset, bool]):  Default: False.
    """

    subject_type: SpendingSubjectType
    subject_id: str
    period: SpendingAlertPeriod
    threshold_cents: int
    webhook_url: Union[Unset, str] = UNSET
    auto_lock: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        subject_type = self.subject_type.value

        subject_id = self.subject_id

        period = self.period.value

        threshold_cents = self.threshold_cents

        webhook_url = self.webhook_url

        auto_lock = self.auto_lock

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "subject_type": subject_type,
                "subject_id": subject_id,
                "period": period,
                "threshold_cents": threshold_cents,
            }
        )
        if webhook_url is not UNSET:
            field_dict["webhook_url"] = webhook_url
        if auto_lock is not UNSET:
            field_dict["auto_lock"] = auto_lock

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        subject_type = SpendingSubjectType(d.pop("subject_type"))

        subject_id = d.pop("subject_id")

        period = SpendingAlertPeriod(d.pop("period"))

        threshold_cents = d.pop("threshold_cents")

        webhook_url = d.pop("webhook_url", UNSET)

        auto_lock = d.pop("auto_lock", UNSET)

        spending_alert_v2_create = cls(
            subject_type=subject_type,
            subject_id=subject_id,
            period=period,
            threshold_cents=threshold_cents,
            webhook_url=webhook_url,
            auto_lock=auto_lock,
        )

        spending_alert_v2_create.additional_properties = d
        return spending_alert_v2_create

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
