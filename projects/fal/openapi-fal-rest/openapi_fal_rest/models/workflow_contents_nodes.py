from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

import attr

if TYPE_CHECKING:
    from ..models.workflow_node import WorkflowNode


T = TypeVar("T", bound="WorkflowContentsNodes")


@attr.s(auto_attribs=True)
class WorkflowContentsNodes:
    """ """

    additional_properties: Dict[str, "WorkflowNode"] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        pass

        field_dict: Dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        field_dict.update({})

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.workflow_node import WorkflowNode

        d = src_dict.copy()
        workflow_contents_nodes = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = WorkflowNode.from_dict(prop_dict)

            additional_properties[prop_name] = additional_property

        workflow_contents_nodes.additional_properties = additional_properties
        return workflow_contents_nodes

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> "WorkflowNode":
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: "WorkflowNode") -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
