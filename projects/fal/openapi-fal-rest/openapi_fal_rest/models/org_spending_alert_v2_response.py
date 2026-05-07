import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.spending_alert_period import SpendingAlertPeriod
from ..models.spending_subject_type import SpendingSubjectType
from ..types import UNSET, Unset

T = TypeVar("T", bound="OrgSpendingAlertV2Response")


@_attrs_define
class OrgSpendingAlertV2Response:
    """
    Attributes:
        id (str):
        user_id (str):
        subject_type (SpendingSubjectType):
        subject_id (str):
        period (SpendingAlertPeriod):
        threshold_cents (int):
        is_enabled (bool):
        auto_lock (bool):
        creator_auth_method (str):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        team_nickname (str):
        last_triggered_at (Union[Unset, datetime.datetime]):
        webhook_url (Union[Unset, str]):
        team_name (Union[Unset, str]):
    """

    id: str
    user_id: str
    subject_type: SpendingSubjectType
    subject_id: str
    period: SpendingAlertPeriod
    threshold_cents: int
    is_enabled: bool
    auto_lock: bool
    creator_auth_method: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    team_nickname: str
    last_triggered_at: Union[Unset, datetime.datetime] = UNSET
    webhook_url: Union[Unset, str] = UNSET
    team_name: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        user_id = self.user_id

        subject_type = self.subject_type.value

        subject_id = self.subject_id

        period = self.period.value

        threshold_cents = self.threshold_cents

        is_enabled = self.is_enabled

        auto_lock = self.auto_lock

        creator_auth_method = self.creator_auth_method

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        team_nickname = self.team_nickname

        last_triggered_at: Union[Unset, str] = UNSET
        if not isinstance(self.last_triggered_at, Unset):
            last_triggered_at = self.last_triggered_at.isoformat()

        webhook_url = self.webhook_url

        team_name = self.team_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "user_id": user_id,
                "subject_type": subject_type,
                "subject_id": subject_id,
                "period": period,
                "threshold_cents": threshold_cents,
                "is_enabled": is_enabled,
                "auto_lock": auto_lock,
                "creator_auth_method": creator_auth_method,
                "created_at": created_at,
                "updated_at": updated_at,
                "team_nickname": team_nickname,
            }
        )
        if last_triggered_at is not UNSET:
            field_dict["last_triggered_at"] = last_triggered_at
        if webhook_url is not UNSET:
            field_dict["webhook_url"] = webhook_url
        if team_name is not UNSET:
            field_dict["team_name"] = team_name

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id")

        user_id = d.pop("user_id")

        subject_type = SpendingSubjectType(d.pop("subject_type"))

        subject_id = d.pop("subject_id")

        period = SpendingAlertPeriod(d.pop("period"))

        threshold_cents = d.pop("threshold_cents")

        is_enabled = d.pop("is_enabled")

        auto_lock = d.pop("auto_lock")

        creator_auth_method = d.pop("creator_auth_method")

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        team_nickname = d.pop("team_nickname")

        _last_triggered_at = d.pop("last_triggered_at", UNSET)
        last_triggered_at: Union[Unset, datetime.datetime]
        if isinstance(_last_triggered_at, Unset):
            last_triggered_at = UNSET
        else:
            last_triggered_at = isoparse(_last_triggered_at)

        webhook_url = d.pop("webhook_url", UNSET)

        team_name = d.pop("team_name", UNSET)

        org_spending_alert_v2_response = cls(
            id=id,
            user_id=user_id,
            subject_type=subject_type,
            subject_id=subject_id,
            period=period,
            threshold_cents=threshold_cents,
            is_enabled=is_enabled,
            auto_lock=auto_lock,
            creator_auth_method=creator_auth_method,
            created_at=created_at,
            updated_at=updated_at,
            team_nickname=team_nickname,
            last_triggered_at=last_triggered_at,
            webhook_url=webhook_url,
            team_name=team_name,
        )

        org_spending_alert_v2_response.additional_properties = d
        return org_spending_alert_v2_response

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
