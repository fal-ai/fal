import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.contract_status import ContractStatus
from ..models.discount_type import DiscountType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.enterprise_contract_discount_terms import EnterpriseContractDiscountTerms


T = TypeVar("T", bound="EnterpriseContract")


@_attrs_define
class EnterpriseContract:
    """
    Attributes:
        contract_id (str):
        user_id (str):
        owner (str):
        status (ContractStatus):
        starts_at (datetime.datetime):
        ends_at (Union[Unset, datetime.datetime]):
        monthly_commitment (Union[Unset, float]):
        total_commitment (Union[Unset, float]):
        discount_type (Union[Unset, DiscountType]):
        discount_terms (Union[Unset, EnterpriseContractDiscountTerms]):
        orb_customer_id (Union[Unset, str]):
    """

    contract_id: str
    user_id: str
    owner: str
    status: ContractStatus
    starts_at: datetime.datetime
    ends_at: Union[Unset, datetime.datetime] = UNSET
    monthly_commitment: Union[Unset, float] = UNSET
    total_commitment: Union[Unset, float] = UNSET
    discount_type: Union[Unset, DiscountType] = UNSET
    discount_terms: Union[Unset, "EnterpriseContractDiscountTerms"] = UNSET
    orb_customer_id: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        contract_id = self.contract_id

        user_id = self.user_id

        owner = self.owner

        status = self.status.value

        starts_at = self.starts_at.isoformat()

        ends_at: Union[Unset, str] = UNSET
        if not isinstance(self.ends_at, Unset):
            ends_at = self.ends_at.isoformat()

        monthly_commitment = self.monthly_commitment

        total_commitment = self.total_commitment

        discount_type: Union[Unset, str] = UNSET
        if not isinstance(self.discount_type, Unset):
            discount_type = self.discount_type.value

        discount_terms: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.discount_terms, Unset):
            discount_terms = self.discount_terms.to_dict()

        orb_customer_id = self.orb_customer_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "contract_id": contract_id,
                "user_id": user_id,
                "owner": owner,
                "status": status,
                "starts_at": starts_at,
            }
        )
        if ends_at is not UNSET:
            field_dict["ends_at"] = ends_at
        if monthly_commitment is not UNSET:
            field_dict["monthly_commitment"] = monthly_commitment
        if total_commitment is not UNSET:
            field_dict["total_commitment"] = total_commitment
        if discount_type is not UNSET:
            field_dict["discount_type"] = discount_type
        if discount_terms is not UNSET:
            field_dict["discount_terms"] = discount_terms
        if orb_customer_id is not UNSET:
            field_dict["orb_customer_id"] = orb_customer_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.enterprise_contract_discount_terms import EnterpriseContractDiscountTerms

        d = src_dict.copy()
        contract_id = d.pop("contract_id")

        user_id = d.pop("user_id")

        owner = d.pop("owner")

        status = ContractStatus(d.pop("status"))

        starts_at = isoparse(d.pop("starts_at"))

        _ends_at = d.pop("ends_at", UNSET)
        ends_at: Union[Unset, datetime.datetime]
        if isinstance(_ends_at, Unset):
            ends_at = UNSET
        else:
            ends_at = isoparse(_ends_at)

        monthly_commitment = d.pop("monthly_commitment", UNSET)

        total_commitment = d.pop("total_commitment", UNSET)

        _discount_type = d.pop("discount_type", UNSET)
        discount_type: Union[Unset, DiscountType]
        if isinstance(_discount_type, Unset):
            discount_type = UNSET
        else:
            discount_type = DiscountType(_discount_type)

        _discount_terms = d.pop("discount_terms", UNSET)
        discount_terms: Union[Unset, EnterpriseContractDiscountTerms]
        if isinstance(_discount_terms, Unset):
            discount_terms = UNSET
        else:
            discount_terms = EnterpriseContractDiscountTerms.from_dict(_discount_terms)

        orb_customer_id = d.pop("orb_customer_id", UNSET)

        enterprise_contract = cls(
            contract_id=contract_id,
            user_id=user_id,
            owner=owner,
            status=status,
            starts_at=starts_at,
            ends_at=ends_at,
            monthly_commitment=monthly_commitment,
            total_commitment=total_commitment,
            discount_type=discount_type,
            discount_terms=discount_terms,
            orb_customer_id=orb_customer_id,
        )

        enterprise_contract.additional_properties = d
        return enterprise_contract

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
