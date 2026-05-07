import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.spending_subject_type import SpendingSubjectType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.org_spending_lock_response_metadata import OrgSpendingLockResponseMetadata


T = TypeVar("T", bound="OrgSpendingLockResponse")


@_attrs_define
class OrgSpendingLockResponse:
    """
    Attributes:
        id (str):
        user_id (str):
        subject_type (SpendingSubjectType):
        subject_id (str):
        reason (str):
        is_active (bool):
        starts_at (datetime.datetime):
        creator_auth_method (str):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        team_nickname (str):
        ends_at (Union[Unset, datetime.datetime]):
        metadata (Union[Unset, OrgSpendingLockResponseMetadata]):
        team_name (Union[Unset, str]):
    """

    id: str
    user_id: str
    subject_type: SpendingSubjectType
    subject_id: str
    reason: str
    is_active: bool
    starts_at: datetime.datetime
    creator_auth_method: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    team_nickname: str
    ends_at: Union[Unset, datetime.datetime] = UNSET
    metadata: Union[Unset, "OrgSpendingLockResponseMetadata"] = UNSET
    team_name: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        user_id = self.user_id

        subject_type = self.subject_type.value

        subject_id = self.subject_id

        reason = self.reason

        is_active = self.is_active

        starts_at = self.starts_at.isoformat()

        creator_auth_method = self.creator_auth_method

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        team_nickname = self.team_nickname

        ends_at: Union[Unset, str] = UNSET
        if not isinstance(self.ends_at, Unset):
            ends_at = self.ends_at.isoformat()

        metadata: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        team_name = self.team_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "user_id": user_id,
                "subject_type": subject_type,
                "subject_id": subject_id,
                "reason": reason,
                "is_active": is_active,
                "starts_at": starts_at,
                "creator_auth_method": creator_auth_method,
                "created_at": created_at,
                "updated_at": updated_at,
                "team_nickname": team_nickname,
            }
        )
        if ends_at is not UNSET:
            field_dict["ends_at"] = ends_at
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if team_name is not UNSET:
            field_dict["team_name"] = team_name

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.org_spending_lock_response_metadata import OrgSpendingLockResponseMetadata

        d = src_dict.copy()
        id = d.pop("id")

        user_id = d.pop("user_id")

        subject_type = SpendingSubjectType(d.pop("subject_type"))

        subject_id = d.pop("subject_id")

        reason = d.pop("reason")

        is_active = d.pop("is_active")

        starts_at = isoparse(d.pop("starts_at"))

        creator_auth_method = d.pop("creator_auth_method")

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        team_nickname = d.pop("team_nickname")

        _ends_at = d.pop("ends_at", UNSET)
        ends_at: Union[Unset, datetime.datetime]
        if isinstance(_ends_at, Unset):
            ends_at = UNSET
        else:
            ends_at = isoparse(_ends_at)

        _metadata = d.pop("metadata", UNSET)
        metadata: Union[Unset, OrgSpendingLockResponseMetadata]
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = OrgSpendingLockResponseMetadata.from_dict(_metadata)

        team_name = d.pop("team_name", UNSET)

        org_spending_lock_response = cls(
            id=id,
            user_id=user_id,
            subject_type=subject_type,
            subject_id=subject_id,
            reason=reason,
            is_active=is_active,
            starts_at=starts_at,
            creator_auth_method=creator_auth_method,
            created_at=created_at,
            updated_at=updated_at,
            team_nickname=team_nickname,
            ends_at=ends_at,
            metadata=metadata,
            team_name=team_name,
        )

        org_spending_lock_response.additional_properties = d
        return org_spending_lock_response

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
