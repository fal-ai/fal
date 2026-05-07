from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.update_user_notification_settings_request_channel_preferences_additional_property import (
        UpdateUserNotificationSettingsRequestChannelPreferencesAdditionalProperty,
    )


T = TypeVar("T", bound="UpdateUserNotificationSettingsRequestChannelPreferences")


@_attrs_define
class UpdateUserNotificationSettingsRequestChannelPreferences:
    """ """

    additional_properties: dict[str, "UpdateUserNotificationSettingsRequestChannelPreferencesAdditionalProperty"] = (
        _attrs_field(init=False, factory=dict)
    )

    def to_dict(self) -> dict[str, Any]:
        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.update_user_notification_settings_request_channel_preferences_additional_property import (
            UpdateUserNotificationSettingsRequestChannelPreferencesAdditionalProperty,
        )

        d = src_dict.copy()
        update_user_notification_settings_request_channel_preferences = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = UpdateUserNotificationSettingsRequestChannelPreferencesAdditionalProperty.from_dict(
                prop_dict
            )

            additional_properties[prop_name] = additional_property

        update_user_notification_settings_request_channel_preferences.additional_properties = additional_properties
        return update_user_notification_settings_request_channel_preferences

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> "UpdateUserNotificationSettingsRequestChannelPreferencesAdditionalProperty":
        return self.additional_properties[key]

    def __setitem__(
        self, key: str, value: "UpdateUserNotificationSettingsRequestChannelPreferencesAdditionalProperty"
    ) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
