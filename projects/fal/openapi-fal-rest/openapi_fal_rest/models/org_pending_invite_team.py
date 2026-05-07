import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.policy_role import PolicyRole
from ..models.team_role import TeamRole
from ..types import UNSET, Unset

T = TypeVar("T", bound="OrgPendingInviteTeam")


@_attrs_define
class OrgPendingInviteTeam:
    """Pending invite info for a team in org user row.

    Attributes:
        team_user_id (str):
        team_nickname (str):
        team_name (str):
        team_role (TeamRole):
        invite_code (str):
        created_at (datetime.datetime):
        policy_role (Union[Unset, PolicyRole]): Team member roles.

            Viewer and Creator serve different purposes and are not strict subsets
            of each other.  From Creator upward tiers are cumulative:
            Creator < Developer < Billing < Admin.
            Must match the DB `policy_role` enum values exactly.
        required_sso_connection (Union[Unset, str]):
    """

    team_user_id: str
    team_nickname: str
    team_name: str
    team_role: TeamRole
    invite_code: str
    created_at: datetime.datetime
    policy_role: Union[Unset, PolicyRole] = UNSET
    required_sso_connection: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        team_user_id = self.team_user_id

        team_nickname = self.team_nickname

        team_name = self.team_name

        team_role = self.team_role.value

        invite_code = self.invite_code

        created_at = self.created_at.isoformat()

        policy_role: Union[Unset, str] = UNSET
        if not isinstance(self.policy_role, Unset):
            policy_role = self.policy_role.value

        required_sso_connection = self.required_sso_connection

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "team_user_id": team_user_id,
                "team_nickname": team_nickname,
                "team_name": team_name,
                "team_role": team_role,
                "invite_code": invite_code,
                "created_at": created_at,
            }
        )
        if policy_role is not UNSET:
            field_dict["policy_role"] = policy_role
        if required_sso_connection is not UNSET:
            field_dict["required_sso_connection"] = required_sso_connection

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        team_user_id = d.pop("team_user_id")

        team_nickname = d.pop("team_nickname")

        team_name = d.pop("team_name")

        team_role = TeamRole(d.pop("team_role"))

        invite_code = d.pop("invite_code")

        created_at = isoparse(d.pop("created_at"))

        _policy_role = d.pop("policy_role", UNSET)
        policy_role: Union[Unset, PolicyRole]
        if isinstance(_policy_role, Unset):
            policy_role = UNSET
        else:
            policy_role = PolicyRole(_policy_role)

        required_sso_connection = d.pop("required_sso_connection", UNSET)

        org_pending_invite_team = cls(
            team_user_id=team_user_id,
            team_nickname=team_nickname,
            team_name=team_name,
            team_role=team_role,
            invite_code=invite_code,
            created_at=created_at,
            policy_role=policy_role,
            required_sso_connection=required_sso_connection,
        )

        org_pending_invite_team.additional_properties = d
        return org_pending_invite_team

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
