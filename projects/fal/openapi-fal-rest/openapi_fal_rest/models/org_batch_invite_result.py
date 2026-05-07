from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.batch_item_status import BatchItemStatus
from ..models.policy_role import PolicyRole
from ..models.team_role import TeamRole
from ..types import UNSET, Unset

T = TypeVar("T", bound="OrgBatchInviteResult")


@_attrs_define
class OrgBatchInviteResult:
    """
    Attributes:
        email (str):
        team_role (TeamRole):
        status (BatchItemStatus):
        team_nickname (str):
        team_name (str):
        team_user_id (str):
        policy_role (Union[Unset, PolicyRole]): Team member roles.

            Viewer and Creator serve different purposes and are not strict subsets
            of each other.  From Creator upward tiers are cumulative:
            Creator < Developer < Billing < Admin.
            Must match the DB `policy_role` enum values exactly.
        invite_id (Union[Unset, str]):
        email_sent (Union[Unset, bool]):
    """

    email: str
    team_role: TeamRole
    status: BatchItemStatus
    team_nickname: str
    team_name: str
    team_user_id: str
    policy_role: Union[Unset, PolicyRole] = UNSET
    invite_id: Union[Unset, str] = UNSET
    email_sent: Union[Unset, bool] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        email = self.email

        team_role = self.team_role.value

        status = self.status.value

        team_nickname = self.team_nickname

        team_name = self.team_name

        team_user_id = self.team_user_id

        policy_role: Union[Unset, str] = UNSET
        if not isinstance(self.policy_role, Unset):
            policy_role = self.policy_role.value

        invite_id = self.invite_id

        email_sent = self.email_sent

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "email": email,
                "team_role": team_role,
                "status": status,
                "team_nickname": team_nickname,
                "team_name": team_name,
                "team_user_id": team_user_id,
            }
        )
        if policy_role is not UNSET:
            field_dict["policy_role"] = policy_role
        if invite_id is not UNSET:
            field_dict["invite_id"] = invite_id
        if email_sent is not UNSET:
            field_dict["email_sent"] = email_sent

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        email = d.pop("email")

        team_role = TeamRole(d.pop("team_role"))

        status = BatchItemStatus(d.pop("status"))

        team_nickname = d.pop("team_nickname")

        team_name = d.pop("team_name")

        team_user_id = d.pop("team_user_id")

        _policy_role = d.pop("policy_role", UNSET)
        policy_role: Union[Unset, PolicyRole]
        if isinstance(_policy_role, Unset):
            policy_role = UNSET
        else:
            policy_role = PolicyRole(_policy_role)

        invite_id = d.pop("invite_id", UNSET)

        email_sent = d.pop("email_sent", UNSET)

        org_batch_invite_result = cls(
            email=email,
            team_role=team_role,
            status=status,
            team_nickname=team_nickname,
            team_name=team_name,
            team_user_id=team_user_id,
            policy_role=policy_role,
            invite_id=invite_id,
            email_sent=email_sent,
        )

        org_batch_invite_result.additional_properties = d
        return org_batch_invite_result

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
