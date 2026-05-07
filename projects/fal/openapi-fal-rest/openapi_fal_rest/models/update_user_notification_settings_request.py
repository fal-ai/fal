from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.update_user_notification_settings_request_channel_preferences import (
        UpdateUserNotificationSettingsRequestChannelPreferences,
    )


T = TypeVar("T", bound="UpdateUserNotificationSettingsRequest")


@_attrs_define
class UpdateUserNotificationSettingsRequest:
    """
    Attributes:
        notification_email (Union[Unset, str]):
        email_notifications_enabled (Union[Unset, bool]):  Default: True.
        channel_preferences (Union[Unset, UpdateUserNotificationSettingsRequestChannelPreferences]):
        queue_size_threshold (Union[Unset, int]):
        queue_size_threshold_duration (Union[Unset, str]):
        http_5xx_threshold (Union[Unset, int]):
        http_5xx_threshold_duration (Union[Unset, str]):
    """

    notification_email: Union[Unset, str] = UNSET
    email_notifications_enabled: Union[Unset, bool] = True
    channel_preferences: Union[Unset, "UpdateUserNotificationSettingsRequestChannelPreferences"] = UNSET
    queue_size_threshold: Union[Unset, int] = UNSET
    queue_size_threshold_duration: Union[Unset, str] = UNSET
    http_5xx_threshold: Union[Unset, int] = UNSET
    http_5xx_threshold_duration: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        notification_email = self.notification_email

        email_notifications_enabled = self.email_notifications_enabled

        channel_preferences: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.channel_preferences, Unset):
            channel_preferences = self.channel_preferences.to_dict()

        queue_size_threshold = self.queue_size_threshold

        queue_size_threshold_duration = self.queue_size_threshold_duration

        http_5xx_threshold = self.http_5xx_threshold

        http_5xx_threshold_duration = self.http_5xx_threshold_duration

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if notification_email is not UNSET:
            field_dict["notification_email"] = notification_email
        if email_notifications_enabled is not UNSET:
            field_dict["email_notifications_enabled"] = email_notifications_enabled
        if channel_preferences is not UNSET:
            field_dict["channel_preferences"] = channel_preferences
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
        from ..models.update_user_notification_settings_request_channel_preferences import (
            UpdateUserNotificationSettingsRequestChannelPreferences,
        )

        d = src_dict.copy()
        notification_email = d.pop("notification_email", UNSET)

        email_notifications_enabled = d.pop("email_notifications_enabled", UNSET)

        _channel_preferences = d.pop("channel_preferences", UNSET)
        channel_preferences: Union[Unset, UpdateUserNotificationSettingsRequestChannelPreferences]
        if isinstance(_channel_preferences, Unset):
            channel_preferences = UNSET
        else:
            channel_preferences = UpdateUserNotificationSettingsRequestChannelPreferences.from_dict(
                _channel_preferences
            )

        queue_size_threshold = d.pop("queue_size_threshold", UNSET)

        queue_size_threshold_duration = d.pop("queue_size_threshold_duration", UNSET)

        http_5xx_threshold = d.pop("http_5xx_threshold", UNSET)

        http_5xx_threshold_duration = d.pop("http_5xx_threshold_duration", UNSET)

        update_user_notification_settings_request = cls(
            notification_email=notification_email,
            email_notifications_enabled=email_notifications_enabled,
            channel_preferences=channel_preferences,
            queue_size_threshold=queue_size_threshold,
            queue_size_threshold_duration=queue_size_threshold_duration,
            http_5xx_threshold=http_5xx_threshold,
            http_5xx_threshold_duration=http_5xx_threshold_duration,
        )

        update_user_notification_settings_request.additional_properties = d
        return update_user_notification_settings_request

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
