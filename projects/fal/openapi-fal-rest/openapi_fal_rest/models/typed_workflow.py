from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.workflow_contents import WorkflowContents


T = TypeVar("T", bound="TypedWorkflow")


@_attrs_define
class TypedWorkflow:
    """
    Attributes:
        name (str):
        title (str):
        contents (WorkflowContents):
        is_public (bool):
        group_id (Union[Unset, str]):
    """

    name: str
    title: str
    contents: "WorkflowContents"
    is_public: bool
    group_id: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        title = self.title

        contents = self.contents.to_dict()

        is_public = self.is_public

        group_id = self.group_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "title": title,
                "contents": contents,
                "is_public": is_public,
            }
        )
        if group_id is not UNSET:
            field_dict["group_id"] = group_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.workflow_contents import WorkflowContents

        d = src_dict.copy()
        name = d.pop("name")

        title = d.pop("title")

        contents = WorkflowContents.from_dict(d.pop("contents"))

        is_public = d.pop("is_public")

        group_id = d.pop("group_id", UNSET)

        typed_workflow = cls(
            name=name,
            title=title,
            contents=contents,
            is_public=is_public,
            group_id=group_id,
        )

        typed_workflow.additional_properties = d
        return typed_workflow

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
