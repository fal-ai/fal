from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="CollaborationTeamInvitePayload")


@_attrs_define
class CollaborationTeamInvitePayload:
    """
    Attributes:
        name (Union[Unset, str]):
        inviter_nickname (Union[Unset, str]):
        inviter_name (Union[Unset, str]):
        invite_code (Union[Unset, str]):
    """

    name: Union[Unset, str] = UNSET
    inviter_nickname: Union[Unset, str] = UNSET
    inviter_name: Union[Unset, str] = UNSET
    invite_code: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        inviter_nickname = self.inviter_nickname

        inviter_name = self.inviter_name

        invite_code = self.invite_code

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if name is not UNSET:
            field_dict["name"] = name
        if inviter_nickname is not UNSET:
            field_dict["inviter_nickname"] = inviter_nickname
        if inviter_name is not UNSET:
            field_dict["inviter_name"] = inviter_name
        if invite_code is not UNSET:
            field_dict["invite_code"] = invite_code

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name", UNSET)

        inviter_nickname = d.pop("inviter_nickname", UNSET)

        inviter_name = d.pop("inviter_name", UNSET)

        invite_code = d.pop("invite_code", UNSET)

        collaboration_team_invite_payload = cls(
            name=name,
            inviter_nickname=inviter_nickname,
            inviter_name=inviter_name,
            invite_code=invite_code,
        )

        collaboration_team_invite_payload.additional_properties = d
        return collaboration_team_invite_payload

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
