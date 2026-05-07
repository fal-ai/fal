from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="PlatformAppModifiedActor")


@_attrs_define
class PlatformAppModifiedActor:
    """
    Attributes:
        user_id (Union[Unset, str]):
        nickname (Union[Unset, str]):
        full_name (Union[Unset, str]):
    """

    user_id: Union[Unset, str] = UNSET
    nickname: Union[Unset, str] = UNSET
    full_name: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        nickname = self.nickname

        full_name = self.full_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if user_id is not UNSET:
            field_dict["user_id"] = user_id
        if nickname is not UNSET:
            field_dict["nickname"] = nickname
        if full_name is not UNSET:
            field_dict["full_name"] = full_name

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id", UNSET)

        nickname = d.pop("nickname", UNSET)

        full_name = d.pop("full_name", UNSET)

        platform_app_modified_actor = cls(
            user_id=user_id,
            nickname=nickname,
            full_name=full_name,
        )

        platform_app_modified_actor.additional_properties = d
        return platform_app_modified_actor

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
