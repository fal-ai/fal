from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

import attr

if TYPE_CHECKING:
    from ..models.comfy_workflow_schema import ComfyWorkflowSchema


T = TypeVar("T", bound="TypedComfyWorkflow")


@attr.s(auto_attribs=True)
class TypedComfyWorkflow:
    """
    Attributes:
        title (str):
        workflow (ComfyWorkflowSchema):
        is_public (bool):
        name (str):
    """

    title: str
    workflow: "ComfyWorkflowSchema"
    is_public: bool
    name: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        title = self.title
        workflow = self.workflow.to_dict()

        is_public = self.is_public
        name = self.name

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "title": title,
                "workflow": workflow,
                "is_public": is_public,
                "name": name,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.comfy_workflow_schema import ComfyWorkflowSchema

        d = src_dict.copy()
        title = d.pop("title")

        workflow = ComfyWorkflowSchema.from_dict(d.pop("workflow"))

        is_public = d.pop("is_public")

        name = d.pop("name")

        typed_comfy_workflow = cls(
            title=title,
            workflow=workflow,
            is_public=is_public,
            name=name,
        )

        typed_comfy_workflow.additional_properties = d
        return typed_comfy_workflow

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
