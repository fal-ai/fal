from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.org_to_orb_hierarchy_sync_task_input import OrgToOrbHierarchySyncTaskInput
    from ..models.platform_discount_sync_task_input import PlatformDiscountSyncTaskInput


T = TypeVar("T", bound="SubmitTaskRunRequest")


@_attrs_define
class SubmitTaskRunRequest:
    """Request to submit a new task run.

    Attributes:
        task (Union['OrgToOrbHierarchySyncTaskInput', 'PlatformDiscountSyncTaskInput']):
    """

    task: Union["OrgToOrbHierarchySyncTaskInput", "PlatformDiscountSyncTaskInput"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.platform_discount_sync_task_input import PlatformDiscountSyncTaskInput

        task: dict[str, Any]
        if isinstance(self.task, PlatformDiscountSyncTaskInput):
            task = self.task.to_dict()
        else:
            task = self.task.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "task": task,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.org_to_orb_hierarchy_sync_task_input import OrgToOrbHierarchySyncTaskInput
        from ..models.platform_discount_sync_task_input import PlatformDiscountSyncTaskInput

        d = src_dict.copy()

        def _parse_task(data: object) -> Union["OrgToOrbHierarchySyncTaskInput", "PlatformDiscountSyncTaskInput"]:
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                task_type_0 = PlatformDiscountSyncTaskInput.from_dict(data)

                return task_type_0
            except:  # noqa: E722
                pass
            if not isinstance(data, dict):
                raise TypeError()
            task_type_1 = OrgToOrbHierarchySyncTaskInput.from_dict(data)

            return task_type_1

        task = _parse_task(d.pop("task"))

        submit_task_run_request = cls(
            task=task,
        )

        submit_task_run_request.additional_properties = d
        return submit_task_run_request

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
