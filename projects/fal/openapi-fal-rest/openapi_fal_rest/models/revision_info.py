import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="RevisionInfo")


@_attrs_define
class RevisionInfo:
    """
    Attributes:
        revision_id (str):
        created_at (datetime.datetime):
        is_current (bool):
        source_image (Union[Unset, str]):
    """

    revision_id: str
    created_at: datetime.datetime
    is_current: bool
    source_image: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        revision_id = self.revision_id

        created_at = self.created_at.isoformat()

        is_current = self.is_current

        source_image = self.source_image

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "revision_id": revision_id,
                "created_at": created_at,
                "is_current": is_current,
            }
        )
        if source_image is not UNSET:
            field_dict["source_image"] = source_image

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        revision_id = d.pop("revision_id")

        created_at = isoparse(d.pop("created_at"))

        is_current = d.pop("is_current")

        source_image = d.pop("source_image", UNSET)

        revision_info = cls(
            revision_id=revision_id,
            created_at=created_at,
            is_current=is_current,
            source_image=source_image,
        )

        revision_info.additional_properties = d
        return revision_info

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
