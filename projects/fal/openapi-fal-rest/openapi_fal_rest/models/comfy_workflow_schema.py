from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.comfy_workflow_schema_extra_data import ComfyWorkflowSchemaExtraData
    from ..models.comfy_workflow_schema_fal_inputs import ComfyWorkflowSchemaFalInputs
    from ..models.comfy_workflow_schema_fal_inputs_dev_info import ComfyWorkflowSchemaFalInputsDevInfo
    from ..models.comfy_workflow_schema_prompt import ComfyWorkflowSchemaPrompt


T = TypeVar("T", bound="ComfyWorkflowSchema")


@attr.s(auto_attribs=True)
class ComfyWorkflowSchema:
    """
    Attributes:
        prompt (ComfyWorkflowSchemaPrompt):
        extra_data (Union[Unset, ComfyWorkflowSchemaExtraData]):
        fal_inputs_dev_info (Union[Unset, ComfyWorkflowSchemaFalInputsDevInfo]):
        fal_inputs (Union[Unset, ComfyWorkflowSchemaFalInputs]):
    """

    prompt: "ComfyWorkflowSchemaPrompt"
    extra_data: Union[Unset, "ComfyWorkflowSchemaExtraData"] = UNSET
    fal_inputs_dev_info: Union[Unset, "ComfyWorkflowSchemaFalInputsDevInfo"] = UNSET
    fal_inputs: Union[Unset, "ComfyWorkflowSchemaFalInputs"] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        prompt = self.prompt.to_dict()

        extra_data: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.extra_data, Unset):
            extra_data = self.extra_data.to_dict()

        fal_inputs_dev_info: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.fal_inputs_dev_info, Unset):
            fal_inputs_dev_info = self.fal_inputs_dev_info.to_dict()

        fal_inputs: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.fal_inputs, Unset):
            fal_inputs = self.fal_inputs.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "prompt": prompt,
            }
        )
        if extra_data is not UNSET:
            field_dict["extra_data"] = extra_data
        if fal_inputs_dev_info is not UNSET:
            field_dict["fal_inputs_dev_info"] = fal_inputs_dev_info
        if fal_inputs is not UNSET:
            field_dict["fal_inputs"] = fal_inputs

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.comfy_workflow_schema_extra_data import ComfyWorkflowSchemaExtraData
        from ..models.comfy_workflow_schema_fal_inputs import ComfyWorkflowSchemaFalInputs
        from ..models.comfy_workflow_schema_fal_inputs_dev_info import ComfyWorkflowSchemaFalInputsDevInfo
        from ..models.comfy_workflow_schema_prompt import ComfyWorkflowSchemaPrompt

        d = src_dict.copy()
        prompt = ComfyWorkflowSchemaPrompt.from_dict(d.pop("prompt"))

        _extra_data = d.pop("extra_data", UNSET)
        extra_data: Union[Unset, ComfyWorkflowSchemaExtraData]
        if isinstance(_extra_data, Unset):
            extra_data = UNSET
        else:
            extra_data = ComfyWorkflowSchemaExtraData.from_dict(_extra_data)

        _fal_inputs_dev_info = d.pop("fal_inputs_dev_info", UNSET)
        fal_inputs_dev_info: Union[Unset, ComfyWorkflowSchemaFalInputsDevInfo]
        if isinstance(_fal_inputs_dev_info, Unset):
            fal_inputs_dev_info = UNSET
        else:
            fal_inputs_dev_info = ComfyWorkflowSchemaFalInputsDevInfo.from_dict(_fal_inputs_dev_info)

        _fal_inputs = d.pop("fal_inputs", UNSET)
        fal_inputs: Union[Unset, ComfyWorkflowSchemaFalInputs]
        if isinstance(_fal_inputs, Unset):
            fal_inputs = UNSET
        else:
            fal_inputs = ComfyWorkflowSchemaFalInputs.from_dict(_fal_inputs)

        comfy_workflow_schema = cls(
            prompt=prompt,
            extra_data=extra_data,
            fal_inputs_dev_info=fal_inputs_dev_info,
            fal_inputs=fal_inputs,
        )

        comfy_workflow_schema.additional_properties = d
        return comfy_workflow_schema

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
