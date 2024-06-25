from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.workflow_contents_metadata import WorkflowContentsMetadata
    from ..models.workflow_contents_nodes import WorkflowContentsNodes
    from ..models.workflow_contents_output import WorkflowContentsOutput
    from ..models.workflow_schema import WorkflowSchema


T = TypeVar("T", bound="WorkflowContents")


@attr.s(auto_attribs=True)
class WorkflowContents:
    """
    Attributes:
        name (str):
        nodes (WorkflowContentsNodes):
        output (WorkflowContentsOutput):
        schema (WorkflowSchema):
        version (str):
        metadata (Union[Unset, WorkflowContentsMetadata]):
    """

    name: str
    nodes: "WorkflowContentsNodes"
    output: "WorkflowContentsOutput"
    schema: "WorkflowSchema"
    version: str
    metadata: Union[Unset, "WorkflowContentsMetadata"] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        nodes = self.nodes.to_dict()

        output = self.output.to_dict()

        schema = self.schema.to_dict()

        version = self.version
        metadata: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "nodes": nodes,
                "output": output,
                "schema": schema,
                "version": version,
            }
        )
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.workflow_contents_metadata import WorkflowContentsMetadata
        from ..models.workflow_contents_nodes import WorkflowContentsNodes
        from ..models.workflow_contents_output import WorkflowContentsOutput
        from ..models.workflow_schema import WorkflowSchema

        d = src_dict.copy()
        name = d.pop("name")

        nodes = WorkflowContentsNodes.from_dict(d.pop("nodes"))

        output = WorkflowContentsOutput.from_dict(d.pop("output"))

        schema = WorkflowSchema.from_dict(d.pop("schema"))

        version = d.pop("version")

        _metadata = d.pop("metadata", UNSET)
        metadata: Union[Unset, WorkflowContentsMetadata]
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = WorkflowContentsMetadata.from_dict(_metadata)

        workflow_contents = cls(
            name=name,
            nodes=nodes,
            output=output,
            schema=schema,
            version=version,
            metadata=metadata,
        )

        workflow_contents.additional_properties = d
        return workflow_contents

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
