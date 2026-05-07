from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.policy_role import PolicyRole
from ..models.team_role import TeamRole
from ..types import UNSET, Unset

T = TypeVar("T", bound="SentUserInvite")


@_attrs_define
class SentUserInvite:
    """Response model for a created invite.

    Attributes:
        code (str):
        invitee_nickname (str):
        user_id (str):
        email_sent (Union[Unset, bool]):  Default: False.
        team_role (Union[Unset, TeamRole]):
        policy_role (Union[Unset, PolicyRole]): Team member roles.

            Viewer and Creator serve different purposes and are not strict subsets
            of each other.  From Creator upward tiers are cumulative:
            Creator < Developer < Billing < Admin.
            Must match the DB `policy_role` enum values exactly.
        team_name (Union[Unset, str]):
        required_sso_connection (Union[Unset, str]):
    """

    code: str
    invitee_nickname: str
    user_id: str
    email_sent: Union[Unset, bool] = False
    team_role: Union[Unset, TeamRole] = UNSET
    policy_role: Union[Unset, PolicyRole] = UNSET
    team_name: Union[Unset, str] = UNSET
    required_sso_connection: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        code = self.code

        invitee_nickname = self.invitee_nickname

        user_id = self.user_id

        email_sent = self.email_sent

        team_role: Union[Unset, str] = UNSET
        if not isinstance(self.team_role, Unset):
            team_role = self.team_role.value

        policy_role: Union[Unset, str] = UNSET
        if not isinstance(self.policy_role, Unset):
            policy_role = self.policy_role.value

        team_name = self.team_name

        required_sso_connection = self.required_sso_connection

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "code": code,
                "invitee_nickname": invitee_nickname,
                "user_id": user_id,
            }
        )
        if email_sent is not UNSET:
            field_dict["email_sent"] = email_sent
        if team_role is not UNSET:
            field_dict["team_role"] = team_role
        if policy_role is not UNSET:
            field_dict["policy_role"] = policy_role
        if team_name is not UNSET:
            field_dict["team_name"] = team_name
        if required_sso_connection is not UNSET:
            field_dict["required_sso_connection"] = required_sso_connection

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        code = d.pop("code")

        invitee_nickname = d.pop("invitee_nickname")

        user_id = d.pop("user_id")

        email_sent = d.pop("email_sent", UNSET)

        _team_role = d.pop("team_role", UNSET)
        team_role: Union[Unset, TeamRole]
        if isinstance(_team_role, Unset):
            team_role = UNSET
        else:
            team_role = TeamRole(_team_role)

        _policy_role = d.pop("policy_role", UNSET)
        policy_role: Union[Unset, PolicyRole]
        if isinstance(_policy_role, Unset):
            policy_role = UNSET
        else:
            policy_role = PolicyRole(_policy_role)

        team_name = d.pop("team_name", UNSET)

        required_sso_connection = d.pop("required_sso_connection", UNSET)

        sent_user_invite = cls(
            code=code,
            invitee_nickname=invitee_nickname,
            user_id=user_id,
            email_sent=email_sent,
            team_role=team_role,
            policy_role=policy_role,
            team_name=team_name,
            required_sso_connection=required_sso_connection,
        )

        sent_user_invite.additional_properties = d
        return sent_user_invite

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
