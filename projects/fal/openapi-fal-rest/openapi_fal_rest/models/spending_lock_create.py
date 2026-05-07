import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.spending_subject_type import SpendingSubjectType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.spending_lock_create_metadata import SpendingLockCreateMetadata


T = TypeVar("T", bound="SpendingLockCreate")


@_attrs_define
class SpendingLockCreate:
    """
    Attributes:
        subject_type (SpendingSubjectType):
        subject_id (str):
        reason (Union[Unset, str]):  Default: 'spending_limit'.
        starts_at (Union[Unset, datetime.datetime]):
        ends_at (Union[Unset, datetime.datetime]):
        metadata (Union[Unset, SpendingLockCreateMetadata]):
    """

    subject_type: SpendingSubjectType
    subject_id: str
    reason: Union[Unset, str] = "spending_limit"
    starts_at: Union[Unset, datetime.datetime] = UNSET
    ends_at: Union[Unset, datetime.datetime] = UNSET
    metadata: Union[Unset, "SpendingLockCreateMetadata"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        subject_type = self.subject_type.value

        subject_id = self.subject_id

        reason = self.reason

        starts_at: Union[Unset, str] = UNSET
        if not isinstance(self.starts_at, Unset):
            starts_at = self.starts_at.isoformat()

        ends_at: Union[Unset, str] = UNSET
        if not isinstance(self.ends_at, Unset):
            ends_at = self.ends_at.isoformat()

        metadata: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "subject_type": subject_type,
                "subject_id": subject_id,
            }
        )
        if reason is not UNSET:
            field_dict["reason"] = reason
        if starts_at is not UNSET:
            field_dict["starts_at"] = starts_at
        if ends_at is not UNSET:
            field_dict["ends_at"] = ends_at
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.spending_lock_create_metadata import SpendingLockCreateMetadata

        d = src_dict.copy()
        subject_type = SpendingSubjectType(d.pop("subject_type"))

        subject_id = d.pop("subject_id")

        reason = d.pop("reason", UNSET)

        _starts_at = d.pop("starts_at", UNSET)
        starts_at: Union[Unset, datetime.datetime]
        if isinstance(_starts_at, Unset):
            starts_at = UNSET
        else:
            starts_at = isoparse(_starts_at)

        _ends_at = d.pop("ends_at", UNSET)
        ends_at: Union[Unset, datetime.datetime]
        if isinstance(_ends_at, Unset):
            ends_at = UNSET
        else:
            ends_at = isoparse(_ends_at)

        _metadata = d.pop("metadata", UNSET)
        metadata: Union[Unset, SpendingLockCreateMetadata]
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = SpendingLockCreateMetadata.from_dict(_metadata)

        spending_lock_create = cls(
            subject_type=subject_type,
            subject_id=subject_id,
            reason=reason,
            starts_at=starts_at,
            ends_at=ends_at,
            metadata=metadata,
        )

        spending_lock_create.additional_properties = d
        return spending_lock_create

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
