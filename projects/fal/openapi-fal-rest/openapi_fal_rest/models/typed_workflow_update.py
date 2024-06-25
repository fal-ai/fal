from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.workflow_contents import WorkflowContents


T = TypeVar("T", bound="TypedWorkflowUpdate")


@attr.s(auto_attribs=True)
class TypedWorkflowUpdate:
    """
    Attributes:
        name (Union[Unset, str]):
        title (Union[Unset, str]):
        contents (Union[Unset, WorkflowContents]):
        is_public (Union[Unset, bool]):
    """

    name: Union[Unset, str] = UNSET
    title: Union[Unset, str] = UNSET
    contents: Union[Unset, "WorkflowContents"] = UNSET
    is_public: Union[Unset, bool] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        title = self.title
        contents: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.contents, Unset):
            contents = self.contents.to_dict()

        is_public = self.is_public

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if name is not UNSET:
            field_dict["name"] = name
        if title is not UNSET:
            field_dict["title"] = title
        if contents is not UNSET:
            field_dict["contents"] = contents
        if is_public is not UNSET:
            field_dict["is_public"] = is_public

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.workflow_contents import WorkflowContents

        d = src_dict.copy()
        name = d.pop("name", UNSET)

        title = d.pop("title", UNSET)

        _contents = d.pop("contents", UNSET)
        contents: Union[Unset, WorkflowContents]
        if isinstance(_contents, Unset):
            contents = UNSET
        else:
            contents = WorkflowContents.from_dict(_contents)

        is_public = d.pop("is_public", UNSET)

        typed_workflow_update = cls(
            name=name,
            title=title,
            contents=contents,
            is_public=is_public,
        )

        typed_workflow_update.additional_properties = d
        return typed_workflow_update

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
