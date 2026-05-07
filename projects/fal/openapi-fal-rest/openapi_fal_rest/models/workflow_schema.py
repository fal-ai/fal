from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.workflow_schema_input import WorkflowSchemaInput
    from ..models.workflow_schema_output import WorkflowSchemaOutput


T = TypeVar("T", bound="WorkflowSchema")


@_attrs_define
class WorkflowSchema:
    """
    Attributes:
        input_ (WorkflowSchemaInput):
        output (WorkflowSchemaOutput):
    """

    input_: "WorkflowSchemaInput"
    output: "WorkflowSchemaOutput"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        input_ = self.input_.to_dict()

        output = self.output.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "input": input_,
                "output": output,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.workflow_schema_input import WorkflowSchemaInput
        from ..models.workflow_schema_output import WorkflowSchemaOutput

        d = src_dict.copy()
        input_ = WorkflowSchemaInput.from_dict(d.pop("input"))

        output = WorkflowSchemaOutput.from_dict(d.pop("output"))

        workflow_schema = cls(
            input_=input_,
            output=output,
        )

        workflow_schema.additional_properties = d
        return workflow_schema

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
