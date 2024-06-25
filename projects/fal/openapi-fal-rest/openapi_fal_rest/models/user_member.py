from typing import Any, Dict, List, Type, TypeVar

import attr

from ..models.team_role import TeamRole

T = TypeVar("T", bound="UserMember")


@attr.s(auto_attribs=True)
class UserMember:
    """
    Attributes:
        auth_id (str):
        nickname (str):
        full_name (str):
        is_owner (bool):
        team_role (TeamRole): An enumeration.
    """

    auth_id: str
    nickname: str
    full_name: str
    is_owner: bool
    team_role: TeamRole
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        auth_id = self.auth_id
        nickname = self.nickname
        full_name = self.full_name
        is_owner = self.is_owner
        team_role = self.team_role.value

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "auth_id": auth_id,
                "nickname": nickname,
                "full_name": full_name,
                "is_owner": is_owner,
                "team_role": team_role,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        auth_id = d.pop("auth_id")

        nickname = d.pop("nickname")

        full_name = d.pop("full_name")

        is_owner = d.pop("is_owner")

        team_role = TeamRole(d.pop("team_role"))

        user_member = cls(
            auth_id=auth_id,
            nickname=nickname,
            full_name=full_name,
            is_owner=is_owner,
            team_role=team_role,
        )

        user_member.additional_properties = d
        return user_member

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
