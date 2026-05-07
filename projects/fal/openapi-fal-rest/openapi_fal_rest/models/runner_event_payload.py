from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="RunnerEventPayload")


@_attrs_define
class RunnerEventPayload:
    """
    Attributes:
        job_id (Union[Unset, str]):
        machine_type (Union[Unset, str]):
        reason (Union[Unset, str]):
    """

    job_id: Union[Unset, str] = UNSET
    machine_type: Union[Unset, str] = UNSET
    reason: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        job_id = self.job_id

        machine_type = self.machine_type

        reason = self.reason

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if job_id is not UNSET:
            field_dict["job_id"] = job_id
        if machine_type is not UNSET:
            field_dict["machine_type"] = machine_type
        if reason is not UNSET:
            field_dict["reason"] = reason

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        job_id = d.pop("job_id", UNSET)

        machine_type = d.pop("machine_type", UNSET)

        reason = d.pop("reason", UNSET)

        runner_event_payload = cls(
            job_id=job_id,
            machine_type=machine_type,
            reason=reason,
        )

        runner_event_payload.additional_properties = d
        return runner_event_payload

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
