import datetime
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.key_policy import KeyPolicy


T = TypeVar("T", bound="PresetKeyInfo")


@_attrs_define
class PresetKeyInfo:
    """
    Attributes:
        user_id (str):
        key_id (str):
        created_at (datetime.datetime):
        creator_auth_method (str):
        policy_preset (str):
        policy (KeyPolicy):
        alias (Union[Unset, str]):  Default: ''.
        description (Union[Unset, str]):  Default: ''.
        expires_at (Union[Unset, datetime.datetime]):
        revoked_at (Union[Unset, datetime.datetime]):
        creator_nickname (Union[Unset, str]):
        creator_email (Union[Unset, str]):
        mode (Union[Literal['preset'], Unset]):  Default: 'preset'.
    """

    user_id: str
    key_id: str
    created_at: datetime.datetime
    creator_auth_method: str
    policy_preset: str
    policy: "KeyPolicy"
    alias: Union[Unset, str] = ""
    description: Union[Unset, str] = ""
    expires_at: Union[Unset, datetime.datetime] = UNSET
    revoked_at: Union[Unset, datetime.datetime] = UNSET
    creator_nickname: Union[Unset, str] = UNSET
    creator_email: Union[Unset, str] = UNSET
    mode: Union[Literal["preset"], Unset] = "preset"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        key_id = self.key_id

        created_at = self.created_at.isoformat()

        creator_auth_method = self.creator_auth_method

        policy_preset = self.policy_preset

        policy = self.policy.to_dict()

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

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "key_id": key_id,
                "created_at": created_at,
                "creator_auth_method": creator_auth_method,
                "policy_preset": policy_preset,
                "policy": policy,
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

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.key_policy import KeyPolicy

        d = src_dict.copy()
        user_id = d.pop("user_id")

        key_id = d.pop("key_id")

        created_at = isoparse(d.pop("created_at"))

        creator_auth_method = d.pop("creator_auth_method")

        policy_preset = d.pop("policy_preset")

        policy = KeyPolicy.from_dict(d.pop("policy"))

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

        mode = cast(Union[Literal["preset"], Unset], d.pop("mode", UNSET))
        if mode != "preset" and not isinstance(mode, Unset):
            raise ValueError(f"mode must match const 'preset', got '{mode}'")

        preset_key_info = cls(
            user_id=user_id,
            key_id=key_id,
            created_at=created_at,
            creator_auth_method=creator_auth_method,
            policy_preset=policy_preset,
            policy=policy,
            alias=alias,
            description=description,
            expires_at=expires_at,
            revoked_at=revoked_at,
            creator_nickname=creator_nickname,
            creator_email=creator_email,
            mode=mode,
        )

        preset_key_info.additional_properties = d
        return preset_key_info

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
