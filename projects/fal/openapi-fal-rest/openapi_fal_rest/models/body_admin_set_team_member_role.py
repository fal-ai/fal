from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.team_role import TeamRole
from ..types import UNSET, Unset

T = TypeVar("T", bound="BodyAdminSetTeamMemberRole")


@_attrs_define
class BodyAdminSetTeamMemberRole:
    """
    Attributes:
        role (TeamRole):
        owner (Union[Unset, bool]):
    """

    role: TeamRole
    owner: Union[Unset, bool] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        role = self.role.value

        owner = self.owner

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "role": role,
            }
        )
        if owner is not UNSET:
            field_dict["owner"] = owner

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        role = TeamRole(d.pop("role"))

        owner = d.pop("owner", UNSET)

        body_admin_set_team_member_role = cls(
            role=role,
            owner=owner,
        )

        body_admin_set_team_member_role.additional_properties = d
        return body_admin_set_team_member_role

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
