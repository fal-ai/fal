from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="BodyAdminUpdateTeamAuthProvider")


@_attrs_define
class BodyAdminUpdateTeamAuthProvider:
    """
    Attributes:
        team_user_str (str):
        auto_control_auth_provider (Union[Unset, str]):
    """

    team_user_str: str
    auto_control_auth_provider: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        team_user_str = self.team_user_str

        auto_control_auth_provider = self.auto_control_auth_provider

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "team_user_str": team_user_str,
            }
        )
        if auto_control_auth_provider is not UNSET:
            field_dict["auto_control_auth_provider"] = auto_control_auth_provider

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        team_user_str = d.pop("team_user_str")

        auto_control_auth_provider = d.pop("auto_control_auth_provider", UNSET)

        body_admin_update_team_auth_provider = cls(
            team_user_str=team_user_str,
            auto_control_auth_provider=auto_control_auth_provider,
        )

        body_admin_update_team_auth_provider.additional_properties = d
        return body_admin_update_team_auth_provider

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
