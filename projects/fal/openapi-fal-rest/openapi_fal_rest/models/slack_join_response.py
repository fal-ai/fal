from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="SlackJoinResponse")


@_attrs_define
class SlackJoinResponse:
    """
    Attributes:
        channel_id (str):
        joined (Union[Unset, bool]):  Default: True.
    """

    channel_id: str
    joined: Union[Unset, bool] = True
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        channel_id = self.channel_id

        joined = self.joined

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "channel_id": channel_id,
            }
        )
        if joined is not UNSET:
            field_dict["joined"] = joined

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        channel_id = d.pop("channel_id")

        joined = d.pop("joined", UNSET)

        slack_join_response = cls(
            channel_id=channel_id,
            joined=joined,
        )

        slack_join_response.additional_properties = d
        return slack_join_response

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
