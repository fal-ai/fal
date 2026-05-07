from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="BasicUserInfo")


@_attrs_define
class BasicUserInfo:
    """
    Attributes:
        user_id (str):
        email (str):
        nickname (str):
        is_personal (bool):
    """

    user_id: str
    email: str
    nickname: str
    is_personal: bool
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        email = self.email

        nickname = self.nickname

        is_personal = self.is_personal

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "email": email,
                "nickname": nickname,
                "is_personal": is_personal,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        email = d.pop("email")

        nickname = d.pop("nickname")

        is_personal = d.pop("is_personal")

        basic_user_info = cls(
            user_id=user_id,
            email=email,
            nickname=nickname,
            is_personal=is_personal,
        )

        basic_user_info.additional_properties = d
        return basic_user_info

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
