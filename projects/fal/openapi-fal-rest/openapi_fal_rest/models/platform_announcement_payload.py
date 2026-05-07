from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="PlatformAnnouncementPayload")


@_attrs_define
class PlatformAnnouncementPayload:
    """
    Attributes:
        subject (Union[Unset, str]):
        message (Union[Unset, str]):
        link (Union[Unset, str]):
    """

    subject: Union[Unset, str] = UNSET
    message: Union[Unset, str] = UNSET
    link: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        subject = self.subject

        message = self.message

        link = self.link

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if subject is not UNSET:
            field_dict["subject"] = subject
        if message is not UNSET:
            field_dict["message"] = message
        if link is not UNSET:
            field_dict["link"] = link

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        subject = d.pop("subject", UNSET)

        message = d.pop("message", UNSET)

        link = d.pop("link", UNSET)

        platform_announcement_payload = cls(
            subject=subject,
            message=message,
            link=link,
        )

        platform_announcement_payload.additional_properties = d
        return platform_announcement_payload

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
