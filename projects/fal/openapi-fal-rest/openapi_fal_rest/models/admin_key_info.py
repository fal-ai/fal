import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.key_scope import KeyScope
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.key_policy import KeyPolicy


T = TypeVar("T", bound="AdminKeyInfo")


@_attrs_define
class AdminKeyInfo:
    """
    Attributes:
        user_id (str):
        key_id (str):
        created_at (datetime.datetime):
        creator_auth_method (str):
        mode (str):
        alias (Union[Unset, str]):  Default: ''.
        description (Union[Unset, str]):  Default: ''.
        expires_at (Union[Unset, datetime.datetime]):
        revoked_at (Union[Unset, datetime.datetime]):
        creator_nickname (Union[Unset, str]):
        creator_email (Union[Unset, str]):
        scope (Union[Unset, KeyScope]):
        policy_preset (Union[Unset, str]):
        policy (Union[Unset, KeyPolicy]):
    """

    user_id: str
    key_id: str
    created_at: datetime.datetime
    creator_auth_method: str
    mode: str
    alias: Union[Unset, str] = ""
    description: Union[Unset, str] = ""
    expires_at: Union[Unset, datetime.datetime] = UNSET
    revoked_at: Union[Unset, datetime.datetime] = UNSET
    creator_nickname: Union[Unset, str] = UNSET
    creator_email: Union[Unset, str] = UNSET
    scope: Union[Unset, KeyScope] = UNSET
    policy_preset: Union[Unset, str] = UNSET
    policy: Union[Unset, "KeyPolicy"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        key_id = self.key_id

        created_at = self.created_at.isoformat()

        creator_auth_method = self.creator_auth_method

        mode = self.mode

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

        scope: Union[Unset, str] = UNSET
        if not isinstance(self.scope, Unset):
            scope = self.scope.value

        policy_preset = self.policy_preset

        policy: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.policy, Unset):
            policy = self.policy.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "key_id": key_id,
                "created_at": created_at,
                "creator_auth_method": creator_auth_method,
                "mode": mode,
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
        if scope is not UNSET:
            field_dict["scope"] = scope
        if policy_preset is not UNSET:
            field_dict["policy_preset"] = policy_preset
        if policy is not UNSET:
            field_dict["policy"] = policy

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.key_policy import KeyPolicy

        d = src_dict.copy()
        user_id = d.pop("user_id")

        key_id = d.pop("key_id")

        created_at = isoparse(d.pop("created_at"))

        creator_auth_method = d.pop("creator_auth_method")

        mode = d.pop("mode")

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

        _scope = d.pop("scope", UNSET)
        scope: Union[Unset, KeyScope]
        if isinstance(_scope, Unset):
            scope = UNSET
        else:
            scope = KeyScope(_scope)

        policy_preset = d.pop("policy_preset", UNSET)

        _policy = d.pop("policy", UNSET)
        policy: Union[Unset, KeyPolicy]
        if isinstance(_policy, Unset):
            policy = UNSET
        else:
            policy = KeyPolicy.from_dict(_policy)

        admin_key_info = cls(
            user_id=user_id,
            key_id=key_id,
            created_at=created_at,
            creator_auth_method=creator_auth_method,
            mode=mode,
            alias=alias,
            description=description,
            expires_at=expires_at,
            revoked_at=revoked_at,
            creator_nickname=creator_nickname,
            creator_email=creator_email,
            scope=scope,
            policy_preset=policy_preset,
            policy=policy,
        )

        admin_key_info.additional_properties = d
        return admin_key_info

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
