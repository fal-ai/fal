from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.comfy_workflow_schema import ComfyWorkflowSchema


T = TypeVar("T", bound="TypedComfyWorkflowUpdate")


@attr.s(auto_attribs=True)
class TypedComfyWorkflowUpdate:
    """
    Attributes:
        title (Union[Unset, str]):
        workflow (Union[Unset, ComfyWorkflowSchema]):
        is_public (Union[Unset, bool]):
        name (Union[Unset, str]):
    """

    title: Union[Unset, str] = UNSET
    workflow: Union[Unset, "ComfyWorkflowSchema"] = UNSET
    is_public: Union[Unset, bool] = UNSET
    name: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        title = self.title
        workflow: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.workflow, Unset):
            workflow = self.workflow.to_dict()

        is_public = self.is_public
        name = self.name

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if title is not UNSET:
            field_dict["title"] = title
        if workflow is not UNSET:
            field_dict["workflow"] = workflow
        if is_public is not UNSET:
            field_dict["is_public"] = is_public
        if name is not UNSET:
            field_dict["name"] = name

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.comfy_workflow_schema import ComfyWorkflowSchema

        d = src_dict.copy()
        title = d.pop("title", UNSET)

        _workflow = d.pop("workflow", UNSET)
        workflow: Union[Unset, ComfyWorkflowSchema]
        if isinstance(_workflow, Unset):
            workflow = UNSET
        else:
            workflow = ComfyWorkflowSchema.from_dict(_workflow)

        is_public = d.pop("is_public", UNSET)

        name = d.pop("name", UNSET)

        typed_comfy_workflow_update = cls(
            title=title,
            workflow=workflow,
            is_public=is_public,
            name=name,
        )

        typed_comfy_workflow_update.additional_properties = d
        return typed_comfy_workflow_update

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
