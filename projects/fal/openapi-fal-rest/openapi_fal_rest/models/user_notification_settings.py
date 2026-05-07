import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.user_notification_settings_channel_defaults import UserNotificationSettingsChannelDefaults
    from ..models.user_notification_settings_channel_preferences import UserNotificationSettingsChannelPreferences


T = TypeVar("T", bound="UserNotificationSettings")


@_attrs_define
class UserNotificationSettings:
    """
    Attributes:
        notification_email (str):
        email_notifications_enabled (bool):
        notification_email_verified_at (Union[Unset, datetime.datetime]):
        channel_preferences (Union[Unset, UserNotificationSettingsChannelPreferences]):
        channel_defaults (Union[Unset, UserNotificationSettingsChannelDefaults]):
        queue_size_threshold (Union[Unset, int]):
        queue_size_threshold_duration (Union[Unset, str]):
        http_5xx_threshold (Union[Unset, int]):
        http_5xx_threshold_duration (Union[Unset, str]):
    """

    notification_email: str
    email_notifications_enabled: bool
    notification_email_verified_at: Union[Unset, datetime.datetime] = UNSET
    channel_preferences: Union[Unset, "UserNotificationSettingsChannelPreferences"] = UNSET
    channel_defaults: Union[Unset, "UserNotificationSettingsChannelDefaults"] = UNSET
    queue_size_threshold: Union[Unset, int] = UNSET
    queue_size_threshold_duration: Union[Unset, str] = UNSET
    http_5xx_threshold: Union[Unset, int] = UNSET
    http_5xx_threshold_duration: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        notification_email = self.notification_email

        email_notifications_enabled = self.email_notifications_enabled

        notification_email_verified_at: Union[Unset, str] = UNSET
        if not isinstance(self.notification_email_verified_at, Unset):
            notification_email_verified_at = self.notification_email_verified_at.isoformat()

        channel_preferences: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.channel_preferences, Unset):
            channel_preferences = self.channel_preferences.to_dict()

        channel_defaults: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.channel_defaults, Unset):
            channel_defaults = self.channel_defaults.to_dict()

        queue_size_threshold = self.queue_size_threshold

        queue_size_threshold_duration = self.queue_size_threshold_duration

        http_5xx_threshold = self.http_5xx_threshold

        http_5xx_threshold_duration = self.http_5xx_threshold_duration

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "notification_email": notification_email,
                "email_notifications_enabled": email_notifications_enabled,
            }
        )
        if notification_email_verified_at is not UNSET:
            field_dict["notification_email_verified_at"] = notification_email_verified_at
        if channel_preferences is not UNSET:
            field_dict["channel_preferences"] = channel_preferences
        if channel_defaults is not UNSET:
            field_dict["channel_defaults"] = channel_defaults
        if queue_size_threshold is not UNSET:
            field_dict["queue_size_threshold"] = queue_size_threshold
        if queue_size_threshold_duration is not UNSET:
            field_dict["queue_size_threshold_duration"] = queue_size_threshold_duration
        if http_5xx_threshold is not UNSET:
            field_dict["http_5xx_threshold"] = http_5xx_threshold
        if http_5xx_threshold_duration is not UNSET:
            field_dict["http_5xx_threshold_duration"] = http_5xx_threshold_duration

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.user_notification_settings_channel_defaults import UserNotificationSettingsChannelDefaults
        from ..models.user_notification_settings_channel_preferences import UserNotificationSettingsChannelPreferences

        d = src_dict.copy()
        notification_email = d.pop("notification_email")

        email_notifications_enabled = d.pop("email_notifications_enabled")

        _notification_email_verified_at = d.pop("notification_email_verified_at", UNSET)
        notification_email_verified_at: Union[Unset, datetime.datetime]
        if isinstance(_notification_email_verified_at, Unset):
            notification_email_verified_at = UNSET
        else:
            notification_email_verified_at = isoparse(_notification_email_verified_at)

        _channel_preferences = d.pop("channel_preferences", UNSET)
        channel_preferences: Union[Unset, UserNotificationSettingsChannelPreferences]
        if isinstance(_channel_preferences, Unset):
            channel_preferences = UNSET
        else:
            channel_preferences = UserNotificationSettingsChannelPreferences.from_dict(_channel_preferences)

        _channel_defaults = d.pop("channel_defaults", UNSET)
        channel_defaults: Union[Unset, UserNotificationSettingsChannelDefaults]
        if isinstance(_channel_defaults, Unset):
            channel_defaults = UNSET
        else:
            channel_defaults = UserNotificationSettingsChannelDefaults.from_dict(_channel_defaults)

        queue_size_threshold = d.pop("queue_size_threshold", UNSET)

        queue_size_threshold_duration = d.pop("queue_size_threshold_duration", UNSET)

        http_5xx_threshold = d.pop("http_5xx_threshold", UNSET)

        http_5xx_threshold_duration = d.pop("http_5xx_threshold_duration", UNSET)

        user_notification_settings = cls(
            notification_email=notification_email,
            email_notifications_enabled=email_notifications_enabled,
            notification_email_verified_at=notification_email_verified_at,
            channel_preferences=channel_preferences,
            channel_defaults=channel_defaults,
            queue_size_threshold=queue_size_threshold,
            queue_size_threshold_duration=queue_size_threshold_duration,
            http_5xx_threshold=http_5xx_threshold,
            http_5xx_threshold_duration=http_5xx_threshold_duration,
        )

        user_notification_settings.additional_properties = d
        return user_notification_settings

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
