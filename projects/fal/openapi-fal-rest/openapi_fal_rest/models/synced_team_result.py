from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="SyncedTeamResult")


@_attrs_define
class SyncedTeamResult:
    """Result of syncing a single team.

    Attributes:
        team_user_id (str):
        team_nickname (str):
        account_type_synced (Union[Unset, bool]):  Default: False.
        model_access_controls_enabled_synced (Union[Unset, bool]):  Default: False.
    """

    team_user_id: str
    team_nickname: str
    account_type_synced: Union[Unset, bool] = False
    model_access_controls_enabled_synced: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        team_user_id = self.team_user_id

        team_nickname = self.team_nickname

        account_type_synced = self.account_type_synced

        model_access_controls_enabled_synced = self.model_access_controls_enabled_synced

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "team_user_id": team_user_id,
                "team_nickname": team_nickname,
            }
        )
        if account_type_synced is not UNSET:
            field_dict["account_type_synced"] = account_type_synced
        if model_access_controls_enabled_synced is not UNSET:
            field_dict["model_access_controls_enabled_synced"] = model_access_controls_enabled_synced

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        team_user_id = d.pop("team_user_id")

        team_nickname = d.pop("team_nickname")

        account_type_synced = d.pop("account_type_synced", UNSET)

        model_access_controls_enabled_synced = d.pop("model_access_controls_enabled_synced", UNSET)

        synced_team_result = cls(
            team_user_id=team_user_id,
            team_nickname=team_nickname,
            account_type_synced=account_type_synced,
            model_access_controls_enabled_synced=model_access_controls_enabled_synced,
        )

        synced_team_result.additional_properties = d
        return synced_team_result

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
