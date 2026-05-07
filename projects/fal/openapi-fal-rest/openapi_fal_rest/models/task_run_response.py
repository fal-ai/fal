import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.task_run_status import TaskRunStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.task_run_response_task_output import TaskRunResponseTaskOutput


T = TypeVar("T", bound="TaskRunResponse")


@_attrs_define
class TaskRunResponse:
    """Response containing task run details.

    Attributes:
        task_run_id (UUID):
        status (TaskRunStatus): Status of a task run.
        task_name (str):
        created_at (datetime.datetime):
        auth_method (str):
        started_at (Union[Unset, datetime.datetime]):
        completed_at (Union[Unset, datetime.datetime]):
        task_output (Union[Unset, TaskRunResponseTaskOutput]):
        error_message (Union[Unset, str]):
    """

    task_run_id: UUID
    status: TaskRunStatus
    task_name: str
    created_at: datetime.datetime
    auth_method: str
    started_at: Union[Unset, datetime.datetime] = UNSET
    completed_at: Union[Unset, datetime.datetime] = UNSET
    task_output: Union[Unset, "TaskRunResponseTaskOutput"] = UNSET
    error_message: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        task_run_id = str(self.task_run_id)

        status = self.status.value

        task_name = self.task_name

        created_at = self.created_at.isoformat()

        auth_method = self.auth_method

        started_at: Union[Unset, str] = UNSET
        if not isinstance(self.started_at, Unset):
            started_at = self.started_at.isoformat()

        completed_at: Union[Unset, str] = UNSET
        if not isinstance(self.completed_at, Unset):
            completed_at = self.completed_at.isoformat()

        task_output: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.task_output, Unset):
            task_output = self.task_output.to_dict()

        error_message = self.error_message

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "task_run_id": task_run_id,
                "status": status,
                "task_name": task_name,
                "created_at": created_at,
                "auth_method": auth_method,
            }
        )
        if started_at is not UNSET:
            field_dict["started_at"] = started_at
        if completed_at is not UNSET:
            field_dict["completed_at"] = completed_at
        if task_output is not UNSET:
            field_dict["task_output"] = task_output
        if error_message is not UNSET:
            field_dict["error_message"] = error_message

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.task_run_response_task_output import TaskRunResponseTaskOutput

        d = src_dict.copy()
        task_run_id = UUID(d.pop("task_run_id"))

        status = TaskRunStatus(d.pop("status"))

        task_name = d.pop("task_name")

        created_at = isoparse(d.pop("created_at"))

        auth_method = d.pop("auth_method")

        _started_at = d.pop("started_at", UNSET)
        started_at: Union[Unset, datetime.datetime]
        if isinstance(_started_at, Unset):
            started_at = UNSET
        else:
            started_at = isoparse(_started_at)

        _completed_at = d.pop("completed_at", UNSET)
        completed_at: Union[Unset, datetime.datetime]
        if isinstance(_completed_at, Unset):
            completed_at = UNSET
        else:
            completed_at = isoparse(_completed_at)

        _task_output = d.pop("task_output", UNSET)
        task_output: Union[Unset, TaskRunResponseTaskOutput]
        if isinstance(_task_output, Unset):
            task_output = UNSET
        else:
            task_output = TaskRunResponseTaskOutput.from_dict(_task_output)

        error_message = d.pop("error_message", UNSET)

        task_run_response = cls(
            task_run_id=task_run_id,
            status=status,
            task_name=task_name,
            created_at=created_at,
            auth_method=auth_method,
            started_at=started_at,
            completed_at=completed_at,
            task_output=task_output,
            error_message=error_message,
        )

        task_run_response.additional_properties = d
        return task_run_response

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
