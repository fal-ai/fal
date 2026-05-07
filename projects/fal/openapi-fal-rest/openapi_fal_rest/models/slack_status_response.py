from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="SlackStatusResponse")


@_attrs_define
class SlackStatusResponse:
    """
    Attributes:
        installed (bool):
        team_id (Union[Unset, str]):
        team_name (Union[Unset, str]):
        bot_user_id (Union[Unset, str]):
        scope (Union[Unset, str]):
        default_channel_id (Union[Unset, str]):
        default_channel_name (Union[Unset, str]):
    """

    installed: bool
    team_id: Union[Unset, str] = UNSET
    team_name: Union[Unset, str] = UNSET
    bot_user_id: Union[Unset, str] = UNSET
    scope: Union[Unset, str] = UNSET
    default_channel_id: Union[Unset, str] = UNSET
    default_channel_name: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        installed = self.installed

        team_id = self.team_id

        team_name = self.team_name

        bot_user_id = self.bot_user_id

        scope = self.scope

        default_channel_id = self.default_channel_id

        default_channel_name = self.default_channel_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "installed": installed,
            }
        )
        if team_id is not UNSET:
            field_dict["team_id"] = team_id
        if team_name is not UNSET:
            field_dict["team_name"] = team_name
        if bot_user_id is not UNSET:
            field_dict["bot_user_id"] = bot_user_id
        if scope is not UNSET:
            field_dict["scope"] = scope
        if default_channel_id is not UNSET:
            field_dict["default_channel_id"] = default_channel_id
        if default_channel_name is not UNSET:
            field_dict["default_channel_name"] = default_channel_name

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        installed = d.pop("installed")

        team_id = d.pop("team_id", UNSET)

        team_name = d.pop("team_name", UNSET)

        bot_user_id = d.pop("bot_user_id", UNSET)

        scope = d.pop("scope", UNSET)

        default_channel_id = d.pop("default_channel_id", UNSET)

        default_channel_name = d.pop("default_channel_name", UNSET)

        slack_status_response = cls(
            installed=installed,
            team_id=team_id,
            team_name=team_name,
            bot_user_id=bot_user_id,
            scope=scope,
            default_channel_id=default_channel_id,
            default_channel_name=default_channel_name,
        )

        slack_status_response.additional_properties = d
        return slack_status_response

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
