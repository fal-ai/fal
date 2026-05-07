from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.user_notification_channel import UserNotificationChannel

if TYPE_CHECKING:
    from ..models.collaboration_team_member_added_payload import CollaborationTeamMemberAddedPayload


T = TypeVar("T", bound="CreateCollaborationTeamMemberAddedNotification")


@_attrs_define
class CreateCollaborationTeamMemberAddedNotification:
    """
    Attributes:
        channels (list[UserNotificationChannel]):
        subcategory (Literal['collaboration_team_member_added']):
        payload (CollaborationTeamMemberAddedPayload):
    """

    channels: list[UserNotificationChannel]
    subcategory: Literal["collaboration_team_member_added"]
    payload: "CollaborationTeamMemberAddedPayload"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        channels = []
        for channels_item_data in self.channels:
            channels_item = channels_item_data.value
            channels.append(channels_item)

        subcategory = self.subcategory

        payload = self.payload.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "channels": channels,
                "subcategory": subcategory,
                "payload": payload,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.collaboration_team_member_added_payload import CollaborationTeamMemberAddedPayload

        d = src_dict.copy()
        channels = []
        _channels = d.pop("channels")
        for channels_item_data in _channels:
            channels_item = UserNotificationChannel(channels_item_data)

            channels.append(channels_item)

        subcategory = cast(Literal["collaboration_team_member_added"], d.pop("subcategory"))
        if subcategory != "collaboration_team_member_added":
            raise ValueError(f"subcategory must match const 'collaboration_team_member_added', got '{subcategory}'")

        payload = CollaborationTeamMemberAddedPayload.from_dict(d.pop("payload"))

        create_collaboration_team_member_added_notification = cls(
            channels=channels,
            subcategory=subcategory,
            payload=payload,
        )

        create_collaboration_team_member_added_notification.additional_properties = d
        return create_collaboration_team_member_added_notification

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
