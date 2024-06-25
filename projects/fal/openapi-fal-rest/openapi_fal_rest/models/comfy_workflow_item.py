import datetime
from typing import Any, Dict, List, Type, TypeVar

import attr
from dateutil.parser import isoparse

T = TypeVar("T", bound="ComfyWorkflowItem")


@attr.s(auto_attribs=True)
class ComfyWorkflowItem:
    """
    Attributes:
        name (str):
        title (str):
        user_id (str):
        created_at (datetime.datetime):
        user_nickname (str):
    """

    name: str
    title: str
    user_id: str
    created_at: datetime.datetime
    user_nickname: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        title = self.title
        user_id = self.user_id
        created_at = self.created_at.isoformat()

        user_nickname = self.user_nickname

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "title": title,
                "user_id": user_id,
                "created_at": created_at,
                "user_nickname": user_nickname,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        title = d.pop("title")

        user_id = d.pop("user_id")

        created_at = isoparse(d.pop("created_at"))

        user_nickname = d.pop("user_nickname")

        comfy_workflow_item = cls(
            name=name,
            title=title,
            user_id=user_id,
            created_at=created_at,
            user_nickname=user_nickname,
        )

        comfy_workflow_item.additional_properties = d
        return comfy_workflow_item

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
