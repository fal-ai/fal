from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="SyncSettingsRequest")


@_attrs_define
class SyncSettingsRequest:
    """Request to sync organization settings to teams.

    Each setting can be individually opted-in for syncing.
    Settings not included or set to False will be skipped.

    Attributes:
        sync_account_type: Sync the org's account_type to all child teams.
        sync_model_access_controls_enabled: Sync whether model access controls
            are enabled (the feature flag), not the actual control entries.

        Attributes:
            sync_account_type (Union[Unset, bool]):  Default: False.
            sync_model_access_controls_enabled (Union[Unset, bool]):  Default: False.
    """

    sync_account_type: Union[Unset, bool] = False
    sync_model_access_controls_enabled: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        sync_account_type = self.sync_account_type

        sync_model_access_controls_enabled = self.sync_model_access_controls_enabled

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if sync_account_type is not UNSET:
            field_dict["sync_account_type"] = sync_account_type
        if sync_model_access_controls_enabled is not UNSET:
            field_dict["sync_model_access_controls_enabled"] = sync_model_access_controls_enabled

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        sync_account_type = d.pop("sync_account_type", UNSET)

        sync_model_access_controls_enabled = d.pop("sync_model_access_controls_enabled", UNSET)

        sync_settings_request = cls(
            sync_account_type=sync_account_type,
            sync_model_access_controls_enabled=sync_model_access_controls_enabled,
        )

        sync_settings_request.additional_properties = d
        return sync_settings_request

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
