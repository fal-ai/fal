import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

import attr
from dateutil.parser import isoparse

if TYPE_CHECKING:
    from ..models.comfy_workflow_schema import ComfyWorkflowSchema


T = TypeVar("T", bound="ComfyWorkflowDetail")


@attr.s(auto_attribs=True)
class ComfyWorkflowDetail:
    """
    Attributes:
        created_at (datetime.datetime):
        user_id (str):
        workflow (ComfyWorkflowSchema):
        is_public (bool):
        title (str):
        name (str):
        user_nickname (str):
    """

    created_at: datetime.datetime
    user_id: str
    workflow: "ComfyWorkflowSchema"
    is_public: bool
    title: str
    name: str
    user_nickname: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        created_at = self.created_at.isoformat()

        user_id = self.user_id
        workflow = self.workflow.to_dict()

        is_public = self.is_public
        title = self.title
        name = self.name
        user_nickname = self.user_nickname

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "created_at": created_at,
                "user_id": user_id,
                "workflow": workflow,
                "is_public": is_public,
                "title": title,
                "name": name,
                "user_nickname": user_nickname,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.comfy_workflow_schema import ComfyWorkflowSchema

        d = src_dict.copy()
        created_at = isoparse(d.pop("created_at"))

        user_id = d.pop("user_id")

        workflow = ComfyWorkflowSchema.from_dict(d.pop("workflow"))

        is_public = d.pop("is_public")

        title = d.pop("title")

        name = d.pop("name")

        user_nickname = d.pop("user_nickname")

        comfy_workflow_detail = cls(
            created_at=created_at,
            user_id=user_id,
            workflow=workflow,
            is_public=is_public,
            title=title,
            name=name,
            user_nickname=user_nickname,
        )

        comfy_workflow_detail.additional_properties = d
        return comfy_workflow_detail

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
