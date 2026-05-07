import datetime
from typing import Any, Literal, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.key_scope import KeyScope
from ..types import UNSET, Unset

T = TypeVar("T", bound="LegacyKeyInfo")


@_attrs_define
class LegacyKeyInfo:
    """
    Attributes:
        user_id (str):
        key_id (str):
        created_at (datetime.datetime):
        creator_auth_method (str):
        alias (Union[Unset, str]):  Default: ''.
        description (Union[Unset, str]):  Default: ''.
        expires_at (Union[Unset, datetime.datetime]):
        revoked_at (Union[Unset, datetime.datetime]):
        creator_nickname (Union[Unset, str]):
        creator_email (Union[Unset, str]):
        mode (Union[Literal['legacy'], Unset]):  Default: 'legacy'.
        scope (Union[Unset, KeyScope]):
    """

    user_id: str
    key_id: str
    created_at: datetime.datetime
    creator_auth_method: str
    alias: Union[Unset, str] = ""
    description: Union[Unset, str] = ""
    expires_at: Union[Unset, datetime.datetime] = UNSET
    revoked_at: Union[Unset, datetime.datetime] = UNSET
    creator_nickname: Union[Unset, str] = UNSET
    creator_email: Union[Unset, str] = UNSET
    mode: Union[Literal["legacy"], Unset] = "legacy"
    scope: Union[Unset, KeyScope] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        key_id = self.key_id

        created_at = self.created_at.isoformat()

        creator_auth_method = self.creator_auth_method

        alias = self.alias

        description = self.description

        expires_at: Union[Unset, str] = UNSET
        if not isinstance(self.expires_at, Unset):
            expires_at = self.expires_at.isoformat()

        revoked_at: Union[Unset, str] = UNSET
        if not isinstance(self.revoked_at, Unset):
            revoked_at = self.revoked_at.isoformat()

        creator_nickname = self.creator_nickname

        creator_email = self.creator_email

        mode = self.mode

        scope: Union[Unset, str] = UNSET
        if not isinstance(self.scope, Unset):
            scope = self.scope.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "key_id": key_id,
                "created_at": created_at,
                "creator_auth_method": creator_auth_method,
            }
        )
        if alias is not UNSET:
            field_dict["alias"] = alias
        if description is not UNSET:
            field_dict["description"] = description
        if expires_at is not UNSET:
            field_dict["expires_at"] = expires_at
        if revoked_at is not UNSET:
            field_dict["revoked_at"] = revoked_at
        if creator_nickname is not UNSET:
            field_dict["creator_nickname"] = creator_nickname
        if creator_email is not UNSET:
            field_dict["creator_email"] = creator_email
        if mode is not UNSET:
            field_dict["mode"] = mode
        if scope is not UNSET:
            field_dict["scope"] = scope

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        key_id = d.pop("key_id")

        created_at = isoparse(d.pop("created_at"))

        creator_auth_method = d.pop("creator_auth_method")

        alias = d.pop("alias", UNSET)

        description = d.pop("description", UNSET)

        _expires_at = d.pop("expires_at", UNSET)
        expires_at: Union[Unset, datetime.datetime]
        if isinstance(_expires_at, Unset):
            expires_at = UNSET
        else:
            expires_at = isoparse(_expires_at)

        _revoked_at = d.pop("revoked_at", UNSET)
        revoked_at: Union[Unset, datetime.datetime]
        if isinstance(_revoked_at, Unset):
            revoked_at = UNSET
        else:
            revoked_at = isoparse(_revoked_at)

        creator_nickname = d.pop("creator_nickname", UNSET)

        creator_email = d.pop("creator_email", UNSET)

        mode = cast(Union[Literal["legacy"], Unset], d.pop("mode", UNSET))
        if mode != "legacy" and not isinstance(mode, Unset):
            raise ValueError(f"mode must match const 'legacy', got '{mode}'")

        _scope = d.pop("scope", UNSET)
        scope: Union[Unset, KeyScope]
        if isinstance(_scope, Unset):
            scope = UNSET
        else:
            scope = KeyScope(_scope)

        legacy_key_info = cls(
            user_id=user_id,
            key_id=key_id,
            created_at=created_at,
            creator_auth_method=creator_auth_method,
            alias=alias,
            description=description,
            expires_at=expires_at,
            revoked_at=revoked_at,
            creator_nickname=creator_nickname,
            creator_email=creator_email,
            mode=mode,
            scope=scope,
        )

        legacy_key_info.additional_properties = d
        return legacy_key_info

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
