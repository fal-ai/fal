from typing import Any, Literal, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.key_scope import KeyScope
from ..types import UNSET, Unset

T = TypeVar("T", bound="KeyAuthMethodInfo")


@_attrs_define
class KeyAuthMethodInfo:
    """Decoration for API key auth methods (key=).

    Attributes:
        key_scope (KeyScope):
        type_ (Union[Literal['key'], Unset]):  Default: 'key'.
        key_alias (Union[Unset, str]):
        key_owner_nickname (Union[Unset, str]):
        key_owner_email (Union[Unset, str]):
        key_creator_nickname (Union[Unset, str]):
        key_creator_email (Union[Unset, str]):
    """

    key_scope: KeyScope
    type_: Union[Literal["key"], Unset] = "key"
    key_alias: Union[Unset, str] = UNSET
    key_owner_nickname: Union[Unset, str] = UNSET
    key_owner_email: Union[Unset, str] = UNSET
    key_creator_nickname: Union[Unset, str] = UNSET
    key_creator_email: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        key_scope = self.key_scope.value

        type_ = self.type_

        key_alias = self.key_alias

        key_owner_nickname = self.key_owner_nickname

        key_owner_email = self.key_owner_email

        key_creator_nickname = self.key_creator_nickname

        key_creator_email = self.key_creator_email

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "key_scope": key_scope,
            }
        )
        if type_ is not UNSET:
            field_dict["type"] = type_
        if key_alias is not UNSET:
            field_dict["key_alias"] = key_alias
        if key_owner_nickname is not UNSET:
            field_dict["key_owner_nickname"] = key_owner_nickname
        if key_owner_email is not UNSET:
            field_dict["key_owner_email"] = key_owner_email
        if key_creator_nickname is not UNSET:
            field_dict["key_creator_nickname"] = key_creator_nickname
        if key_creator_email is not UNSET:
            field_dict["key_creator_email"] = key_creator_email

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        key_scope = KeyScope(d.pop("key_scope"))

        type_ = cast(Union[Literal["key"], Unset], d.pop("type", UNSET))
        if type_ != "key" and not isinstance(type_, Unset):
            raise ValueError(f"type must match const 'key', got '{type_}'")

        key_alias = d.pop("key_alias", UNSET)

        key_owner_nickname = d.pop("key_owner_nickname", UNSET)

        key_owner_email = d.pop("key_owner_email", UNSET)

        key_creator_nickname = d.pop("key_creator_nickname", UNSET)

        key_creator_email = d.pop("key_creator_email", UNSET)

        key_auth_method_info = cls(
            key_scope=key_scope,
            type_=type_,
            key_alias=key_alias,
            key_owner_nickname=key_owner_nickname,
            key_owner_email=key_owner_email,
            key_creator_nickname=key_creator_nickname,
            key_creator_email=key_creator_email,
        )

        key_auth_method_info.additional_properties = d
        return key_auth_method_info

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
