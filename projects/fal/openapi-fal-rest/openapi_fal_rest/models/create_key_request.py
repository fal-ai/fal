import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.policy_preset import PolicyPreset
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.key_policy import KeyPolicy


T = TypeVar("T", bound="CreateKeyRequest")


@_attrs_define
class CreateKeyRequest:
    """
    Attributes:
        preset (Union[Unset, PolicyPreset]): Permission presets for API keys.

            Must match the DB `policy_preset` enum values exactly.
        policy (Union[Unset, KeyPolicy]):
        alias (Union[Unset, str]):  Default: ''.
        description (Union[Unset, str]):  Default: ''.
        expires_at (Union[Unset, datetime.datetime]):
    """

    preset: Union[Unset, PolicyPreset] = UNSET
    policy: Union[Unset, "KeyPolicy"] = UNSET
    alias: Union[Unset, str] = ""
    description: Union[Unset, str] = ""
    expires_at: Union[Unset, datetime.datetime] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        preset: Union[Unset, str] = UNSET
        if not isinstance(self.preset, Unset):
            preset = self.preset.value

        policy: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.policy, Unset):
            policy = self.policy.to_dict()

        alias = self.alias

        description = self.description

        expires_at: Union[Unset, str] = UNSET
        if not isinstance(self.expires_at, Unset):
            expires_at = self.expires_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if preset is not UNSET:
            field_dict["preset"] = preset
        if policy is not UNSET:
            field_dict["policy"] = policy
        if alias is not UNSET:
            field_dict["alias"] = alias
        if description is not UNSET:
            field_dict["description"] = description
        if expires_at is not UNSET:
            field_dict["expires_at"] = expires_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.key_policy import KeyPolicy

        d = src_dict.copy()
        _preset = d.pop("preset", UNSET)
        preset: Union[Unset, PolicyPreset]
        if isinstance(_preset, Unset):
            preset = UNSET
        else:
            preset = PolicyPreset(_preset)

        _policy = d.pop("policy", UNSET)
        policy: Union[Unset, KeyPolicy]
        if isinstance(_policy, Unset):
            policy = UNSET
        else:
            policy = KeyPolicy.from_dict(_policy)

        alias = d.pop("alias", UNSET)

        description = d.pop("description", UNSET)

        _expires_at = d.pop("expires_at", UNSET)
        expires_at: Union[Unset, datetime.datetime]
        if isinstance(_expires_at, Unset):
            expires_at = UNSET
        else:
            expires_at = isoparse(_expires_at)

        create_key_request = cls(
            preset=preset,
            policy=policy,
            alias=alias,
            description=description,
            expires_at=expires_at,
        )

        create_key_request.additional_properties = d
        return create_key_request

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
