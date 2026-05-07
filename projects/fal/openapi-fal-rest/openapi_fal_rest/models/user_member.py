from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.policy_role import PolicyRole
from ..models.team_role import TeamRole
from ..types import UNSET, Unset

T = TypeVar("T", bound="UserMember")


@_attrs_define
class UserMember:
    """
    Attributes:
        auth_id (str):
        nickname (str):
        email (str):
        full_name (str):
        is_owner (bool):
        team_role (TeamRole):
        workos_sub (Union[Unset, str]):
        all_auth_methods (Union[Unset, list[str]]):
        policy_role (Union[Unset, PolicyRole]): Team member roles.

            Viewer and Creator serve different purposes and are not strict subsets
            of each other.  From Creator upward tiers are cumulative:
            Creator < Developer < Billing < Admin.
            Must match the DB `policy_role` enum values exactly.
    """

    auth_id: str
    nickname: str
    email: str
    full_name: str
    is_owner: bool
    team_role: TeamRole
    workos_sub: Union[Unset, str] = UNSET
    all_auth_methods: Union[Unset, list[str]] = UNSET
    policy_role: Union[Unset, PolicyRole] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        auth_id = self.auth_id

        nickname = self.nickname

        email = self.email

        full_name = self.full_name

        is_owner = self.is_owner

        team_role = self.team_role.value

        workos_sub = self.workos_sub

        all_auth_methods: Union[Unset, list[str]] = UNSET
        if not isinstance(self.all_auth_methods, Unset):
            all_auth_methods = self.all_auth_methods

        policy_role: Union[Unset, str] = UNSET
        if not isinstance(self.policy_role, Unset):
            policy_role = self.policy_role.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "auth_id": auth_id,
                "nickname": nickname,
                "email": email,
                "full_name": full_name,
                "is_owner": is_owner,
                "team_role": team_role,
            }
        )
        if workos_sub is not UNSET:
            field_dict["workos_sub"] = workos_sub
        if all_auth_methods is not UNSET:
            field_dict["all_auth_methods"] = all_auth_methods
        if policy_role is not UNSET:
            field_dict["policy_role"] = policy_role

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        auth_id = d.pop("auth_id")

        nickname = d.pop("nickname")

        email = d.pop("email")

        full_name = d.pop("full_name")

        is_owner = d.pop("is_owner")

        team_role = TeamRole(d.pop("team_role"))

        workos_sub = d.pop("workos_sub", UNSET)

        all_auth_methods = cast(list[str], d.pop("all_auth_methods", UNSET))

        _policy_role = d.pop("policy_role", UNSET)
        policy_role: Union[Unset, PolicyRole]
        if isinstance(_policy_role, Unset):
            policy_role = UNSET
        else:
            policy_role = PolicyRole(_policy_role)

        user_member = cls(
            auth_id=auth_id,
            nickname=nickname,
            email=email,
            full_name=full_name,
            is_owner=is_owner,
            team_role=team_role,
            workos_sub=workos_sub,
            all_auth_methods=all_auth_methods,
            policy_role=policy_role,
        )

        user_member.additional_properties = d
        return user_member

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
