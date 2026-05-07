from typing import Any, TypeVar, Union
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="TagAssignmentBulkSet")


@_attrs_define
class TagAssignmentBulkSet:
    """
    Attributes:
        entity_type (str):
        entity_id (str):
        tag_ids (Union[Unset, list[UUID]]):
    """

    entity_type: str
    entity_id: str
    tag_ids: Union[Unset, list[UUID]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        entity_type = self.entity_type

        entity_id = self.entity_id

        tag_ids: Union[Unset, list[str]] = UNSET
        if not isinstance(self.tag_ids, Unset):
            tag_ids = []
            for tag_ids_item_data in self.tag_ids:
                tag_ids_item = str(tag_ids_item_data)
                tag_ids.append(tag_ids_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "entity_type": entity_type,
                "entity_id": entity_id,
            }
        )
        if tag_ids is not UNSET:
            field_dict["tag_ids"] = tag_ids

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        entity_type = d.pop("entity_type")

        entity_id = d.pop("entity_id")

        tag_ids = []
        _tag_ids = d.pop("tag_ids", UNSET)
        for tag_ids_item_data in _tag_ids or []:
            tag_ids_item = UUID(tag_ids_item_data)

            tag_ids.append(tag_ids_item)

        tag_assignment_bulk_set = cls(
            entity_type=entity_type,
            entity_id=entity_id,
            tag_ids=tag_ids,
        )

        tag_assignment_bulk_set.additional_properties = d
        return tag_assignment_bulk_set

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
