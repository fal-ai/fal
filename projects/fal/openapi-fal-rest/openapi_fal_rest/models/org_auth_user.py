import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="OrgAuthUser")


@_attrs_define
class OrgAuthUser:
    """An auth user entry for an org user row.

    Attributes:
        auth_id (str):
        full_name (Union[Unset, str]):
        personal_user_id (Union[Unset, str]):
        personal_user_nickname (Union[Unset, str]):
        sso_connection (Union[Unset, str]):
        last_seen_at (Union[Unset, datetime.datetime]):
    """

    auth_id: str
    full_name: Union[Unset, str] = UNSET
    personal_user_id: Union[Unset, str] = UNSET
    personal_user_nickname: Union[Unset, str] = UNSET
    sso_connection: Union[Unset, str] = UNSET
    last_seen_at: Union[Unset, datetime.datetime] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        auth_id = self.auth_id

        full_name = self.full_name

        personal_user_id = self.personal_user_id

        personal_user_nickname = self.personal_user_nickname

        sso_connection = self.sso_connection

        last_seen_at: Union[Unset, str] = UNSET
        if not isinstance(self.last_seen_at, Unset):
            last_seen_at = self.last_seen_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "auth_id": auth_id,
            }
        )
        if full_name is not UNSET:
            field_dict["full_name"] = full_name
        if personal_user_id is not UNSET:
            field_dict["personal_user_id"] = personal_user_id
        if personal_user_nickname is not UNSET:
            field_dict["personal_user_nickname"] = personal_user_nickname
        if sso_connection is not UNSET:
            field_dict["sso_connection"] = sso_connection
        if last_seen_at is not UNSET:
            field_dict["last_seen_at"] = last_seen_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        auth_id = d.pop("auth_id")

        full_name = d.pop("full_name", UNSET)

        personal_user_id = d.pop("personal_user_id", UNSET)

        personal_user_nickname = d.pop("personal_user_nickname", UNSET)

        sso_connection = d.pop("sso_connection", UNSET)

        _last_seen_at = d.pop("last_seen_at", UNSET)
        last_seen_at: Union[Unset, datetime.datetime]
        if isinstance(_last_seen_at, Unset):
            last_seen_at = UNSET
        else:
            last_seen_at = isoparse(_last_seen_at)

        org_auth_user = cls(
            auth_id=auth_id,
            full_name=full_name,
            personal_user_id=personal_user_id,
            personal_user_nickname=personal_user_nickname,
            sso_connection=sso_connection,
            last_seen_at=last_seen_at,
        )

        org_auth_user.additional_properties = d
        return org_auth_user

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
