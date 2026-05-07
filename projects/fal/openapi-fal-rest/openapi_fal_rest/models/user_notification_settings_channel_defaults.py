from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.user_notification_settings_channel_defaults_additional_property import (
        UserNotificationSettingsChannelDefaultsAdditionalProperty,
    )


T = TypeVar("T", bound="UserNotificationSettingsChannelDefaults")


@_attrs_define
class UserNotificationSettingsChannelDefaults:
    """ """

    additional_properties: dict[str, "UserNotificationSettingsChannelDefaultsAdditionalProperty"] = _attrs_field(
        init=False, factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.user_notification_settings_channel_defaults_additional_property import (
            UserNotificationSettingsChannelDefaultsAdditionalProperty,
        )

        d = src_dict.copy()
        user_notification_settings_channel_defaults = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = UserNotificationSettingsChannelDefaultsAdditionalProperty.from_dict(prop_dict)

            additional_properties[prop_name] = additional_property

        user_notification_settings_channel_defaults.additional_properties = additional_properties
        return user_notification_settings_channel_defaults

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> "UserNotificationSettingsChannelDefaultsAdditionalProperty":
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: "UserNotificationSettingsChannelDefaultsAdditionalProperty") -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
