import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.key_scope import KeyScope
from ..types import UNSET, Unset

T = TypeVar("T", bound="AdminUserKeyInfo")


@_attrs_define
class AdminUserKeyInfo:
    """
    Attributes:
        user_id (str):
        key_id (str):
        created_at (datetime.datetime):
        scope (KeyScope):
        alias (str):
        creator_auth_method (str):
        creator_nickname (Union[Unset, str]):
        creator_email (Union[Unset, str]):
    """

    user_id: str
    key_id: str
    created_at: datetime.datetime
    scope: KeyScope
    alias: str
    creator_auth_method: str
    creator_nickname: Union[Unset, str] = UNSET
    creator_email: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        key_id = self.key_id

        created_at = self.created_at.isoformat()

        scope = self.scope.value

        alias = self.alias

        creator_auth_method = self.creator_auth_method

        creator_nickname = self.creator_nickname

        creator_email = self.creator_email

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "key_id": key_id,
                "created_at": created_at,
                "scope": scope,
                "alias": alias,
                "creator_auth_method": creator_auth_method,
            }
        )
        if creator_nickname is not UNSET:
            field_dict["creator_nickname"] = creator_nickname
        if creator_email is not UNSET:
            field_dict["creator_email"] = creator_email

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        key_id = d.pop("key_id")

        created_at = isoparse(d.pop("created_at"))

        scope = KeyScope(d.pop("scope"))

        alias = d.pop("alias")

        creator_auth_method = d.pop("creator_auth_method")

        creator_nickname = d.pop("creator_nickname", UNSET)

        creator_email = d.pop("creator_email", UNSET)

        admin_user_key_info = cls(
            user_id=user_id,
            key_id=key_id,
            created_at=created_at,
            scope=scope,
            alias=alias,
            creator_auth_method=creator_auth_method,
            creator_nickname=creator_nickname,
            creator_email=creator_email,
        )

        admin_user_key_info.additional_properties = d
        return admin_user_key_info

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
