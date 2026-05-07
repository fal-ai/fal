from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.app_notification_settings_channel_preferences_override_additional_property import (
        AppNotificationSettingsChannelPreferencesOverrideAdditionalProperty,
    )


T = TypeVar("T", bound="AppNotificationSettingsChannelPreferencesOverride")


@_attrs_define
class AppNotificationSettingsChannelPreferencesOverride:
    """ """

    additional_properties: dict[str, "AppNotificationSettingsChannelPreferencesOverrideAdditionalProperty"] = (
        _attrs_field(init=False, factory=dict)
    )

    def to_dict(self) -> dict[str, Any]:
        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.app_notification_settings_channel_preferences_override_additional_property import (
            AppNotificationSettingsChannelPreferencesOverrideAdditionalProperty,
        )

        d = src_dict.copy()
        app_notification_settings_channel_preferences_override = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = AppNotificationSettingsChannelPreferencesOverrideAdditionalProperty.from_dict(
                prop_dict
            )

            additional_properties[prop_name] = additional_property

        app_notification_settings_channel_preferences_override.additional_properties = additional_properties
        return app_notification_settings_channel_preferences_override

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> "AppNotificationSettingsChannelPreferencesOverrideAdditionalProperty":
        return self.additional_properties[key]

    def __setitem__(
        self, key: str, value: "AppNotificationSettingsChannelPreferencesOverrideAdditionalProperty"
    ) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
