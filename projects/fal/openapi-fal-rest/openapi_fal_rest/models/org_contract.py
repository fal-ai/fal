import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="OrgContract")


@_attrs_define
class OrgContract:
    """Enterprise contract with user_id for organization context.

    Attributes:
        contract_id (str):
        starts_at (datetime.datetime):
        user_id (str):
        ends_at (Union[Unset, datetime.datetime]):
        total_commitment (Union[Unset, float]):
        monthly_commitment (Union[Unset, float]):
    """

    contract_id: str
    starts_at: datetime.datetime
    user_id: str
    ends_at: Union[Unset, datetime.datetime] = UNSET
    total_commitment: Union[Unset, float] = UNSET
    monthly_commitment: Union[Unset, float] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        contract_id = self.contract_id

        starts_at = self.starts_at.isoformat()

        user_id = self.user_id

        ends_at: Union[Unset, str] = UNSET
        if not isinstance(self.ends_at, Unset):
            ends_at = self.ends_at.isoformat()

        total_commitment = self.total_commitment

        monthly_commitment = self.monthly_commitment

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "contract_id": contract_id,
                "starts_at": starts_at,
                "user_id": user_id,
            }
        )
        if ends_at is not UNSET:
            field_dict["ends_at"] = ends_at
        if total_commitment is not UNSET:
            field_dict["total_commitment"] = total_commitment
        if monthly_commitment is not UNSET:
            field_dict["monthly_commitment"] = monthly_commitment

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        contract_id = d.pop("contract_id")

        starts_at = isoparse(d.pop("starts_at"))

        user_id = d.pop("user_id")

        _ends_at = d.pop("ends_at", UNSET)
        ends_at: Union[Unset, datetime.datetime]
        if isinstance(_ends_at, Unset):
            ends_at = UNSET
        else:
            ends_at = isoparse(_ends_at)

        total_commitment = d.pop("total_commitment", UNSET)

        monthly_commitment = d.pop("monthly_commitment", UNSET)

        org_contract = cls(
            contract_id=contract_id,
            starts_at=starts_at,
            user_id=user_id,
            ends_at=ends_at,
            total_commitment=total_commitment,
            monthly_commitment=monthly_commitment,
        )

        org_contract.additional_properties = d
        return org_contract

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
