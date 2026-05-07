from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.policy_role import PolicyRole
from ..models.team_role import TeamRole
from ..types import UNSET, Unset

T = TypeVar("T", bound="UserInvite")


@_attrs_define
class UserInvite:
    """Request model for creating an invite.

    Attributes:
        invitee_nickname (str):
        team_role (TeamRole):
        policy_role (Union[Unset, PolicyRole]): Team member roles.

            Viewer and Creator serve different purposes and are not strict subsets
            of each other.  From Creator upward tiers are cumulative:
            Creator < Developer < Billing < Admin.
            Must match the DB `policy_role` enum values exactly.
        required_sso_connection (Union[Unset, str]):
    """

    invitee_nickname: str
    team_role: TeamRole
    policy_role: Union[Unset, PolicyRole] = UNSET
    required_sso_connection: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        invitee_nickname = self.invitee_nickname

        team_role = self.team_role.value

        policy_role: Union[Unset, str] = UNSET
        if not isinstance(self.policy_role, Unset):
            policy_role = self.policy_role.value

        required_sso_connection = self.required_sso_connection

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "invitee_nickname": invitee_nickname,
                "team_role": team_role,
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
        invitee_nickname = d.pop("invitee_nickname")

        team_role = TeamRole(d.pop("team_role"))

        _policy_role = d.pop("policy_role", UNSET)
        policy_role: Union[Unset, PolicyRole]
        if isinstance(_policy_role, Unset):
            policy_role = UNSET
        else:
            policy_role = PolicyRole(_policy_role)

        required_sso_connection = d.pop("required_sso_connection", UNSET)

        user_invite = cls(
            invitee_nickname=invitee_nickname,
            team_role=team_role,
            policy_role=policy_role,
            required_sso_connection=required_sso_connection,
        )

        user_invite.additional_properties = d
        return user_invite

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
