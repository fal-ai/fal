import datetime
from typing import Any, Literal, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="OrgToOrbHierarchySyncTaskInput")


@_attrs_define
class OrgToOrbHierarchySyncTaskInput:
    """Input for org-to-orb-hierarchy-sync Faktory task.

    Given any `user_id` in an org, resolves the org and syncs all of its
    child teams into the org's Orb hierarchy: attaches each child to the
    parent customer, adds them to the parent subscription's
    `usage_customer_ids` on every active plan-default usage interval, and
    end-dates each child's standalone subscription.

        Attributes:
            user_id (str):
            task_name (Union[Literal['billing_org_to_orb_hierarchy_sync'], Unset]):  Default:
                'billing_org_to_orb_hierarchy_sync'.
            dry_run (Union[Unset, bool]):  Default: False.
            bypass_reasons (Union[Unset, list[str]]):
            effective_date (Union[Unset, datetime.date]):
            detach_orphaned_children (Union[Unset, bool]):  Default: False.
    """

    user_id: str
    task_name: Union[Literal["billing_org_to_orb_hierarchy_sync"], Unset] = "billing_org_to_orb_hierarchy_sync"
    dry_run: Union[Unset, bool] = False
    bypass_reasons: Union[Unset, list[str]] = UNSET
    effective_date: Union[Unset, datetime.date] = UNSET
    detach_orphaned_children: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        task_name = self.task_name

        dry_run = self.dry_run

        bypass_reasons: Union[Unset, list[str]] = UNSET
        if not isinstance(self.bypass_reasons, Unset):
            bypass_reasons = self.bypass_reasons

        effective_date: Union[Unset, str] = UNSET
        if not isinstance(self.effective_date, Unset):
            effective_date = self.effective_date.isoformat()

        detach_orphaned_children = self.detach_orphaned_children

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
            }
        )
        if task_name is not UNSET:
            field_dict["task_name"] = task_name
        if dry_run is not UNSET:
            field_dict["dry_run"] = dry_run
        if bypass_reasons is not UNSET:
            field_dict["bypass_reasons"] = bypass_reasons
        if effective_date is not UNSET:
            field_dict["effective_date"] = effective_date
        if detach_orphaned_children is not UNSET:
            field_dict["detach_orphaned_children"] = detach_orphaned_children

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        task_name = cast(Union[Literal["billing_org_to_orb_hierarchy_sync"], Unset], d.pop("task_name", UNSET))
        if task_name != "billing_org_to_orb_hierarchy_sync" and not isinstance(task_name, Unset):
            raise ValueError(f"task_name must match const 'billing_org_to_orb_hierarchy_sync', got '{task_name}'")

        dry_run = d.pop("dry_run", UNSET)

        bypass_reasons = cast(list[str], d.pop("bypass_reasons", UNSET))

        _effective_date = d.pop("effective_date", UNSET)
        effective_date: Union[Unset, datetime.date]
        if isinstance(_effective_date, Unset):
            effective_date = UNSET
        else:
            effective_date = isoparse(_effective_date).date()

        detach_orphaned_children = d.pop("detach_orphaned_children", UNSET)

        org_to_orb_hierarchy_sync_task_input = cls(
            user_id=user_id,
            task_name=task_name,
            dry_run=dry_run,
            bypass_reasons=bypass_reasons,
            effective_date=effective_date,
            detach_orphaned_children=detach_orphaned_children,
        )

        org_to_orb_hierarchy_sync_task_input.additional_properties = d
        return org_to_orb_hierarchy_sync_task_input

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
