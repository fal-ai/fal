from typing import Any, Literal, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="PlatformDiscountSyncTaskInput")


@_attrs_define
class PlatformDiscountSyncTaskInput:
    """Input for platform discount sync Faktory task.

    Attributes:
        org_user_id (str):
        task_name (Union[Literal['billing_platform_discount_sync'], Unset]):  Default: 'billing_platform_discount_sync'.
        dry_run (Union[Unset, bool]):  Default: False.
    """

    org_user_id: str
    task_name: Union[Literal["billing_platform_discount_sync"], Unset] = "billing_platform_discount_sync"
    dry_run: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        org_user_id = self.org_user_id

        task_name = self.task_name

        dry_run = self.dry_run

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "org_user_id": org_user_id,
            }
        )
        if task_name is not UNSET:
            field_dict["task_name"] = task_name
        if dry_run is not UNSET:
            field_dict["dry_run"] = dry_run

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        org_user_id = d.pop("org_user_id")

        task_name = cast(Union[Literal["billing_platform_discount_sync"], Unset], d.pop("task_name", UNSET))
        if task_name != "billing_platform_discount_sync" and not isinstance(task_name, Unset):
            raise ValueError(f"task_name must match const 'billing_platform_discount_sync', got '{task_name}'")

        dry_run = d.pop("dry_run", UNSET)

        platform_discount_sync_task_input = cls(
            org_user_id=org_user_id,
            task_name=task_name,
            dry_run=dry_run,
        )

        platform_discount_sync_task_input.additional_properties = d
        return platform_discount_sync_task_input

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
