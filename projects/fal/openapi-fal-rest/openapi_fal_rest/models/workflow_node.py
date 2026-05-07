from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.workflow_node_type import WorkflowNodeType

T = TypeVar("T", bound="WorkflowNode")


@_attrs_define
class WorkflowNode:
    """
    Attributes:
        type_ (WorkflowNodeType):
        id (str):
        depends (list[str]):
    """

    type_: WorkflowNodeType
    id: str
    depends: list[str]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_.value

        id = self.id

        depends = self.depends

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type_,
                "id": id,
                "depends": depends,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        type_ = WorkflowNodeType(d.pop("type"))

        id = d.pop("id")

        depends = cast(list[str], d.pop("depends"))

        workflow_node = cls(
            type_=type_,
            id=id,
            depends=depends,
        )

        workflow_node.additional_properties = d
        return workflow_node

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
