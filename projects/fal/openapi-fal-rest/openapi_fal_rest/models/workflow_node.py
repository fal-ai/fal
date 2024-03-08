from typing import Any, Dict, List, Type, TypeVar, cast

import attr

from ..models.workflow_node_type import WorkflowNodeType

T = TypeVar("T", bound="WorkflowNode")


@attr.s(auto_attribs=True)
class WorkflowNode:
    """
    Attributes:
        type (WorkflowNodeType):
        id (str):
        depends (List[str]):
    """

    type: WorkflowNodeType
    id: str
    depends: List[str]
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        type = self.type.value

        id = self.id
        depends = self.depends

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type,
                "id": id,
                "depends": depends,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        type = WorkflowNodeType(d.pop("type"))

        id = d.pop("id")

        depends = cast(List[str], d.pop("depends"))

        workflow_node = cls(
            type=type,
            id=id,
            depends=depends,
        )

        workflow_node.additional_properties = d
        return workflow_node

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
