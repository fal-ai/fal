import datetime
from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="WorkflowItem")


@_attrs_define
class WorkflowItem:
    """
    Attributes:
        name (str):
        title (str):
        user_id (str):
        user_nickname (str):
        created_at (datetime.datetime):
        tags (list[str]):
        endpoints (list[str]):
        thumbnail_url (Union[Unset, str]):
        description (Union[Unset, str]):
        group_id (Union[Unset, str]):
    """

    name: str
    title: str
    user_id: str
    user_nickname: str
    created_at: datetime.datetime
    tags: list[str]
    endpoints: list[str]
    thumbnail_url: Union[Unset, str] = UNSET
    description: Union[Unset, str] = UNSET
    group_id: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        title = self.title

        user_id = self.user_id

        user_nickname = self.user_nickname

        created_at = self.created_at.isoformat()

        tags = self.tags

        endpoints = self.endpoints

        thumbnail_url = self.thumbnail_url

        description = self.description

        group_id = self.group_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "title": title,
                "user_id": user_id,
                "user_nickname": user_nickname,
                "created_at": created_at,
                "tags": tags,
                "endpoints": endpoints,
            }
        )
        if thumbnail_url is not UNSET:
            field_dict["thumbnail_url"] = thumbnail_url
        if description is not UNSET:
            field_dict["description"] = description
        if group_id is not UNSET:
            field_dict["group_id"] = group_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        title = d.pop("title")

        user_id = d.pop("user_id")

        user_nickname = d.pop("user_nickname")

        created_at = isoparse(d.pop("created_at"))

        tags = cast(list[str], d.pop("tags"))

        endpoints = cast(list[str], d.pop("endpoints"))

        thumbnail_url = d.pop("thumbnail_url", UNSET)

        description = d.pop("description", UNSET)

        group_id = d.pop("group_id", UNSET)

        workflow_item = cls(
            name=name,
            title=title,
            user_id=user_id,
            user_nickname=user_nickname,
            created_at=created_at,
            tags=tags,
            endpoints=endpoints,
            thumbnail_url=thumbnail_url,
            description=description,
            group_id=group_id,
        )

        workflow_item.additional_properties = d
        return workflow_item

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
