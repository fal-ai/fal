from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.account_type import AccountType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.synced_team_result import SyncedTeamResult


T = TypeVar("T", bound="SyncSettingsResponse")


@_attrs_define
class SyncSettingsResponse:
    """Response from sync settings action.

    Attributes:
        teams_updated (int):
        teams_skipped (int):
        results (list['SyncedTeamResult']):
        synced_account_type (Union[Unset, AccountType]):
        synced_model_access_controls_enabled (Union[Unset, bool]):
    """

    teams_updated: int
    teams_skipped: int
    results: list["SyncedTeamResult"]
    synced_account_type: Union[Unset, AccountType] = UNSET
    synced_model_access_controls_enabled: Union[Unset, bool] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        teams_updated = self.teams_updated

        teams_skipped = self.teams_skipped

        results = []
        for results_item_data in self.results:
            results_item = results_item_data.to_dict()
            results.append(results_item)

        synced_account_type: Union[Unset, str] = UNSET
        if not isinstance(self.synced_account_type, Unset):
            synced_account_type = self.synced_account_type.value

        synced_model_access_controls_enabled = self.synced_model_access_controls_enabled

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "teams_updated": teams_updated,
                "teams_skipped": teams_skipped,
                "results": results,
            }
        )
        if synced_account_type is not UNSET:
            field_dict["synced_account_type"] = synced_account_type
        if synced_model_access_controls_enabled is not UNSET:
            field_dict["synced_model_access_controls_enabled"] = synced_model_access_controls_enabled

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.synced_team_result import SyncedTeamResult

        d = src_dict.copy()
        teams_updated = d.pop("teams_updated")

        teams_skipped = d.pop("teams_skipped")

        results = []
        _results = d.pop("results")
        for results_item_data in _results:
            results_item = SyncedTeamResult.from_dict(results_item_data)

            results.append(results_item)

        _synced_account_type = d.pop("synced_account_type", UNSET)
        synced_account_type: Union[Unset, AccountType]
        if isinstance(_synced_account_type, Unset):
            synced_account_type = UNSET
        else:
            synced_account_type = AccountType(_synced_account_type)

        synced_model_access_controls_enabled = d.pop("synced_model_access_controls_enabled", UNSET)

        sync_settings_response = cls(
            teams_updated=teams_updated,
            teams_skipped=teams_skipped,
            results=results,
            synced_account_type=synced_account_type,
            synced_model_access_controls_enabled=synced_model_access_controls_enabled,
        )

        sync_settings_response.additional_properties = d
        return sync_settings_response

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
