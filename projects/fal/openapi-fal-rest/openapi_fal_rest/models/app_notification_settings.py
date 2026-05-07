from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.app_notification_settings_channel_preferences_override import (
        AppNotificationSettingsChannelPreferencesOverride,
    )


T = TypeVar("T", bound="AppNotificationSettings")


@_attrs_define
class AppNotificationSettings:
    """
    Attributes:
        queue_size_threshold (Union[Unset, int]):
        queue_size_threshold_duration (Union[Unset, str]):
        http_5xx_threshold (Union[Unset, int]):
        http_5xx_threshold_duration (Union[Unset, str]):
        channel_preferences_override (Union[Unset, AppNotificationSettingsChannelPreferencesOverride]):
    """

    queue_size_threshold: Union[Unset, int] = UNSET
    queue_size_threshold_duration: Union[Unset, str] = UNSET
    http_5xx_threshold: Union[Unset, int] = UNSET
    http_5xx_threshold_duration: Union[Unset, str] = UNSET
    channel_preferences_override: Union[Unset, "AppNotificationSettingsChannelPreferencesOverride"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        queue_size_threshold = self.queue_size_threshold

        queue_size_threshold_duration = self.queue_size_threshold_duration

        http_5xx_threshold = self.http_5xx_threshold

        http_5xx_threshold_duration = self.http_5xx_threshold_duration

        channel_preferences_override: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.channel_preferences_override, Unset):
            channel_preferences_override = self.channel_preferences_override.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if queue_size_threshold is not UNSET:
            field_dict["queue_size_threshold"] = queue_size_threshold
        if queue_size_threshold_duration is not UNSET:
            field_dict["queue_size_threshold_duration"] = queue_size_threshold_duration
        if http_5xx_threshold is not UNSET:
            field_dict["http_5xx_threshold"] = http_5xx_threshold
        if http_5xx_threshold_duration is not UNSET:
            field_dict["http_5xx_threshold_duration"] = http_5xx_threshold_duration
        if channel_preferences_override is not UNSET:
            field_dict["channel_preferences_override"] = channel_preferences_override

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.app_notification_settings_channel_preferences_override import (
            AppNotificationSettingsChannelPreferencesOverride,
        )

        d = src_dict.copy()
        queue_size_threshold = d.pop("queue_size_threshold", UNSET)

        queue_size_threshold_duration = d.pop("queue_size_threshold_duration", UNSET)

        http_5xx_threshold = d.pop("http_5xx_threshold", UNSET)

        http_5xx_threshold_duration = d.pop("http_5xx_threshold_duration", UNSET)

        _channel_preferences_override = d.pop("channel_preferences_override", UNSET)
        channel_preferences_override: Union[Unset, AppNotificationSettingsChannelPreferencesOverride]
        if isinstance(_channel_preferences_override, Unset):
            channel_preferences_override = UNSET
        else:
            channel_preferences_override = AppNotificationSettingsChannelPreferencesOverride.from_dict(
                _channel_preferences_override
            )

        app_notification_settings = cls(
            queue_size_threshold=queue_size_threshold,
            queue_size_threshold_duration=queue_size_threshold_duration,
            http_5xx_threshold=http_5xx_threshold,
            http_5xx_threshold_duration=http_5xx_threshold_duration,
            channel_preferences_override=channel_preferences_override,
        )

        app_notification_settings.additional_properties = d
        return app_notification_settings

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
