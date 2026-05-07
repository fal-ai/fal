from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.policy_role import PolicyRole
from ..models.team_role import TeamRole
from ..types import UNSET, Unset

T = TypeVar("T", bound="OrgUserTeamAssignment")


@_attrs_define
class OrgUserTeamAssignment:
    """A team assignment for an org user (auth user -> team membership).

    Attributes:
        auth_id (str):
        team_user_id (str):
        team_nickname (str):
        team_name (str):
        role (TeamRole):
        is_org_team (bool):
        is_owner (bool):
        team_auto_control_auth_provider (Union[Unset, str]):
        policy_role (Union[Unset, PolicyRole]): Team member roles.

            Viewer and Creator serve different purposes and are not strict subsets
            of each other.  From Creator upward tiers are cumulative:
            Creator < Developer < Billing < Admin.
            Must match the DB `policy_role` enum values exactly.
    """

    auth_id: str
    team_user_id: str
    team_nickname: str
    team_name: str
    role: TeamRole
    is_org_team: bool
    is_owner: bool
    team_auto_control_auth_provider: Union[Unset, str] = UNSET
    policy_role: Union[Unset, PolicyRole] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        auth_id = self.auth_id

        team_user_id = self.team_user_id

        team_nickname = self.team_nickname

        team_name = self.team_name

        role = self.role.value

        is_org_team = self.is_org_team

        is_owner = self.is_owner

        team_auto_control_auth_provider = self.team_auto_control_auth_provider

        policy_role: Union[Unset, str] = UNSET
        if not isinstance(self.policy_role, Unset):
            policy_role = self.policy_role.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "auth_id": auth_id,
                "team_user_id": team_user_id,
                "team_nickname": team_nickname,
                "team_name": team_name,
                "role": role,
                "is_org_team": is_org_team,
                "is_owner": is_owner,
            }
        )
        if team_auto_control_auth_provider is not UNSET:
            field_dict["team_auto_control_auth_provider"] = team_auto_control_auth_provider
        if policy_role is not UNSET:
            field_dict["policy_role"] = policy_role

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        auth_id = d.pop("auth_id")

        team_user_id = d.pop("team_user_id")

        team_nickname = d.pop("team_nickname")

        team_name = d.pop("team_name")

        role = TeamRole(d.pop("role"))

        is_org_team = d.pop("is_org_team")

        is_owner = d.pop("is_owner")

        team_auto_control_auth_provider = d.pop("team_auto_control_auth_provider", UNSET)

        _policy_role = d.pop("policy_role", UNSET)
        policy_role: Union[Unset, PolicyRole]
        if isinstance(_policy_role, Unset):
            policy_role = UNSET
        else:
            policy_role = PolicyRole(_policy_role)

        org_user_team_assignment = cls(
            auth_id=auth_id,
            team_user_id=team_user_id,
            team_nickname=team_nickname,
            team_name=team_name,
            role=role,
            is_org_team=is_org_team,
            is_owner=is_owner,
            team_auto_control_auth_provider=team_auto_control_auth_provider,
            policy_role=policy_role,
        )

        org_user_team_assignment.additional_properties = d
        return org_user_team_assignment

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
