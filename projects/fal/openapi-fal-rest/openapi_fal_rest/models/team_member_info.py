from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.policy_role import PolicyRole
from ..models.team_role import TeamRole
from ..types import UNSET, Unset

T = TypeVar("T", bound="TeamMemberInfo")


@_attrs_define
class TeamMemberInfo:
    """Information about a team member (auth user).

    Attributes:
        auth_id (str):
        email (str):
        full_name (str):
        is_owner (bool):
        team_role (TeamRole):
        personal_user_nickname (Union[Unset, str]):
        policy_role (Union[Unset, PolicyRole]): Team member roles.

            Viewer and Creator serve different purposes and are not strict subsets
            of each other.  From Creator upward tiers are cumulative:
            Creator < Developer < Billing < Admin.
            Must match the DB `policy_role` enum values exactly.
    """

    auth_id: str
    email: str
    full_name: str
    is_owner: bool
    team_role: TeamRole
    personal_user_nickname: Union[Unset, str] = UNSET
    policy_role: Union[Unset, PolicyRole] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        auth_id = self.auth_id

        email = self.email

        full_name = self.full_name

        is_owner = self.is_owner

        team_role = self.team_role.value

        personal_user_nickname = self.personal_user_nickname

        policy_role: Union[Unset, str] = UNSET
        if not isinstance(self.policy_role, Unset):
            policy_role = self.policy_role.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "auth_id": auth_id,
                "email": email,
                "full_name": full_name,
                "is_owner": is_owner,
                "team_role": team_role,
            }
        )
        if personal_user_nickname is not UNSET:
            field_dict["personal_user_nickname"] = personal_user_nickname
        if policy_role is not UNSET:
            field_dict["policy_role"] = policy_role

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        auth_id = d.pop("auth_id")

        email = d.pop("email")

        full_name = d.pop("full_name")

        is_owner = d.pop("is_owner")

        team_role = TeamRole(d.pop("team_role"))

        personal_user_nickname = d.pop("personal_user_nickname", UNSET)

        _policy_role = d.pop("policy_role", UNSET)
        policy_role: Union[Unset, PolicyRole]
        if isinstance(_policy_role, Unset):
            policy_role = UNSET
        else:
            policy_role = PolicyRole(_policy_role)

        team_member_info = cls(
            auth_id=auth_id,
            email=email,
            full_name=full_name,
            is_owner=is_owner,
            team_role=team_role,
            personal_user_nickname=personal_user_nickname,
            policy_role=policy_role,
        )

        team_member_info.additional_properties = d
        return team_member_info

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
