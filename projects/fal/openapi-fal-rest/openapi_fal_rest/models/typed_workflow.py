from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

import attr

if TYPE_CHECKING:
    from ..models.workflow_contents import WorkflowContents


T = TypeVar("T", bound="TypedWorkflow")


@attr.s(auto_attribs=True)
class TypedWorkflow:
    """
    Attributes:
        name (str):
        title (str):
        contents (WorkflowContents):
        is_public (bool):
    """

    name: str
    title: str
    contents: "WorkflowContents"
    is_public: bool
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        title = self.title
        contents = self.contents.to_dict()

        is_public = self.is_public

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "title": title,
                "contents": contents,
                "is_public": is_public,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.workflow_contents import WorkflowContents

        d = src_dict.copy()
        name = d.pop("name")

        title = d.pop("title")

        contents = WorkflowContents.from_dict(d.pop("contents"))

        is_public = d.pop("is_public")

        typed_workflow = cls(
            name=name,
            title=title,
            contents=contents,
            is_public=is_public,
        )

        typed_workflow.additional_properties = d
        return typed_workflow

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
