from typing import Any, Literal, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="UserAuthMethodInfo")


@_attrs_define
class UserAuthMethodInfo:
    """Decoration for user auth methods (workos=, auth0=).

    Attributes:
        user_email (str):
        type_ (Union[Literal['user'], Unset]):  Default: 'user'.
        user_nickname (Union[Unset, str]):
    """

    user_email: str
    type_: Union[Literal["user"], Unset] = "user"
    user_nickname: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_email = self.user_email

        type_ = self.type_

        user_nickname = self.user_nickname

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_email": user_email,
            }
        )
        if type_ is not UNSET:
            field_dict["type"] = type_
        if user_nickname is not UNSET:
            field_dict["user_nickname"] = user_nickname

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_email = d.pop("user_email")

        type_ = cast(Union[Literal["user"], Unset], d.pop("type", UNSET))
        if type_ != "user" and not isinstance(type_, Unset):
            raise ValueError(f"type must match const 'user', got '{type_}'")

        user_nickname = d.pop("user_nickname", UNSET)

        user_auth_method_info = cls(
            user_email=user_email,
            type_=type_,
            user_nickname=user_nickname,
        )

        user_auth_method_info.additional_properties = d
        return user_auth_method_info

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
