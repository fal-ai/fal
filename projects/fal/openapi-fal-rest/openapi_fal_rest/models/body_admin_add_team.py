from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.team_user import TeamUser


T = TypeVar("T", bound="BodyAdminAddTeam")


@_attrs_define
class BodyAdminAddTeam:
    """
    Attributes:
        team_user (TeamUser):
        auto_control_auth_provider (Union[Unset, str]):
        admin_user_str (Union[Unset, str]):
    """

    team_user: "TeamUser"
    auto_control_auth_provider: Union[Unset, str] = UNSET
    admin_user_str: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        team_user = self.team_user.to_dict()

        auto_control_auth_provider = self.auto_control_auth_provider

        admin_user_str = self.admin_user_str

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "team_user": team_user,
            }
        )
        if auto_control_auth_provider is not UNSET:
            field_dict["auto_control_auth_provider"] = auto_control_auth_provider
        if admin_user_str is not UNSET:
            field_dict["admin_user_str"] = admin_user_str

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.team_user import TeamUser

        d = src_dict.copy()
        team_user = TeamUser.from_dict(d.pop("team_user"))

        auto_control_auth_provider = d.pop("auto_control_auth_provider", UNSET)

        admin_user_str = d.pop("admin_user_str", UNSET)

        body_admin_add_team = cls(
            team_user=team_user,
            auto_control_auth_provider=auto_control_auth_provider,
            admin_user_str=admin_user_str,
        )

        body_admin_add_team.additional_properties = d
        return body_admin_add_team

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
