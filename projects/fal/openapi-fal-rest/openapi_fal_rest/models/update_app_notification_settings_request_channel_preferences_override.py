from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.update_app_notification_settings_request_channel_preferences_override_additional_property import (
        UpdateAppNotificationSettingsRequestChannelPreferencesOverrideAdditionalProperty,
    )


T = TypeVar("T", bound="UpdateAppNotificationSettingsRequestChannelPreferencesOverride")


@_attrs_define
class UpdateAppNotificationSettingsRequestChannelPreferencesOverride:
    """ """

    additional_properties: dict[
        str, "UpdateAppNotificationSettingsRequestChannelPreferencesOverrideAdditionalProperty"
    ] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.update_app_notification_settings_request_channel_preferences_override_additional_property import (
            UpdateAppNotificationSettingsRequestChannelPreferencesOverrideAdditionalProperty,
        )

        d = src_dict.copy()
        update_app_notification_settings_request_channel_preferences_override = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = (
                UpdateAppNotificationSettingsRequestChannelPreferencesOverrideAdditionalProperty.from_dict(prop_dict)
            )

            additional_properties[prop_name] = additional_property

        update_app_notification_settings_request_channel_preferences_override.additional_properties = (
            additional_properties
        )
        return update_app_notification_settings_request_channel_preferences_override

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(
        self, key: str
    ) -> "UpdateAppNotificationSettingsRequestChannelPreferencesOverrideAdditionalProperty":
        return self.additional_properties[key]

    def __setitem__(
        self, key: str, value: "UpdateAppNotificationSettingsRequestChannelPreferencesOverrideAdditionalProperty"
    ) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
