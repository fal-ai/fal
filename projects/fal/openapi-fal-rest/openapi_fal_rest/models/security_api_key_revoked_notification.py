import datetime
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.user_notification_channel import UserNotificationChannel

if TYPE_CHECKING:
    from ..models.security_api_key_revoked_payload import SecurityApiKeyRevokedPayload


T = TypeVar("T", bound="SecurityApiKeyRevokedNotification")


@_attrs_define
class SecurityApiKeyRevokedNotification:
    """
    Attributes:
        notification_id (UUID):
        created_at (datetime.datetime):
        category (str):
        user_id (Union[None, str]):
        channels (list[UserNotificationChannel]):
        is_pending (bool):
        group_id (Union[None, str]):
        subcategory (Literal['security_api_key_revoked']):
        payload (SecurityApiKeyRevokedPayload):
    """

    notification_id: UUID
    created_at: datetime.datetime
    category: str
    user_id: Union[None, str]
    channels: list[UserNotificationChannel]
    is_pending: bool
    group_id: Union[None, str]
    subcategory: Literal["security_api_key_revoked"]
    payload: "SecurityApiKeyRevokedPayload"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        notification_id = str(self.notification_id)

        created_at = self.created_at.isoformat()

        category = self.category

        user_id: Union[None, str]
        user_id = self.user_id

        channels = []
        for channels_item_data in self.channels:
            channels_item = channels_item_data.value
            channels.append(channels_item)

        is_pending = self.is_pending

        group_id: Union[None, str]
        group_id = self.group_id

        subcategory = self.subcategory

        payload = self.payload.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "notification_id": notification_id,
                "created_at": created_at,
                "category": category,
                "user_id": user_id,
                "channels": channels,
                "is_pending": is_pending,
                "group_id": group_id,
                "subcategory": subcategory,
                "payload": payload,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.security_api_key_revoked_payload import SecurityApiKeyRevokedPayload

        d = src_dict.copy()
        notification_id = UUID(d.pop("notification_id"))

        created_at = isoparse(d.pop("created_at"))

        category = d.pop("category")

        def _parse_user_id(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        user_id = _parse_user_id(d.pop("user_id"))

        channels = []
        _channels = d.pop("channels")
        for channels_item_data in _channels:
            channels_item = UserNotificationChannel(channels_item_data)

            channels.append(channels_item)

        is_pending = d.pop("is_pending")

        def _parse_group_id(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        group_id = _parse_group_id(d.pop("group_id"))

        subcategory = cast(Literal["security_api_key_revoked"], d.pop("subcategory"))
        if subcategory != "security_api_key_revoked":
            raise ValueError(f"subcategory must match const 'security_api_key_revoked', got '{subcategory}'")

        payload = SecurityApiKeyRevokedPayload.from_dict(d.pop("payload"))

        security_api_key_revoked_notification = cls(
            notification_id=notification_id,
            created_at=created_at,
            category=category,
            user_id=user_id,
            channels=channels,
            is_pending=is_pending,
            group_id=group_id,
            subcategory=subcategory,
            payload=payload,
        )

        security_api_key_revoked_notification.additional_properties = d
        return security_api_key_revoked_notification

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
