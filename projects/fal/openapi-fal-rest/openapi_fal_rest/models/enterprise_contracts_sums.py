from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="EnterpriseContractsSums")


@_attrs_define
class EnterpriseContractsSums:
    """
    Attributes:
        sum_total_commitment (Union[Unset, float]):
        sum_monthly_commitment (Union[Unset, float]):
        owners (Union[Unset, list[str]]):
    """

    sum_total_commitment: Union[Unset, float] = UNSET
    sum_monthly_commitment: Union[Unset, float] = UNSET
    owners: Union[Unset, list[str]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        sum_total_commitment = self.sum_total_commitment

        sum_monthly_commitment = self.sum_monthly_commitment

        owners: Union[Unset, list[str]] = UNSET
        if not isinstance(self.owners, Unset):
            owners = self.owners

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if sum_total_commitment is not UNSET:
            field_dict["sum_total_commitment"] = sum_total_commitment
        if sum_monthly_commitment is not UNSET:
            field_dict["sum_monthly_commitment"] = sum_monthly_commitment
        if owners is not UNSET:
            field_dict["owners"] = owners

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        sum_total_commitment = d.pop("sum_total_commitment", UNSET)

        sum_monthly_commitment = d.pop("sum_monthly_commitment", UNSET)

        owners = cast(list[str], d.pop("owners", UNSET))

        enterprise_contracts_sums = cls(
            sum_total_commitment=sum_total_commitment,
            sum_monthly_commitment=sum_monthly_commitment,
            owners=owners,
        )

        enterprise_contracts_sums.additional_properties = d
        return enterprise_contracts_sums

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
