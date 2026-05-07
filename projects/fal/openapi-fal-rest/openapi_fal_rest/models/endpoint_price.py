from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.endpoint_provider_type import EndpointProviderType
from ..models.enterprise_status import EnterpriseStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="EndpointPrice")


@_attrs_define
class EndpointPrice:
    """
    Attributes:
        endpoint (str):
        price (float):
        billable_unit (str):
        provider (EndpointProviderType):
        in_registry (Union[Unset, bool]):
        enterprise_status (Union[Unset, EnterpriseStatus]): Enterprise readiness status for endpoints.

            READY: Endpoint meets enterprise compliance standards (DPA, no training on data)
            PENDING: Endpoint is under review or does not yet meet enterprise standards
    """

    endpoint: str
    price: float
    billable_unit: str
    provider: EndpointProviderType
    in_registry: Union[Unset, bool] = UNSET
    enterprise_status: Union[Unset, EnterpriseStatus] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        endpoint = self.endpoint

        price = self.price

        billable_unit = self.billable_unit

        provider = self.provider.value

        in_registry = self.in_registry

        enterprise_status: Union[Unset, str] = UNSET
        if not isinstance(self.enterprise_status, Unset):
            enterprise_status = self.enterprise_status.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "endpoint": endpoint,
                "price": price,
                "billable_unit": billable_unit,
                "provider": provider,
            }
        )
        if in_registry is not UNSET:
            field_dict["in_registry"] = in_registry
        if enterprise_status is not UNSET:
            field_dict["enterprise_status"] = enterprise_status

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        endpoint = d.pop("endpoint")

        price = d.pop("price")

        billable_unit = d.pop("billable_unit")

        provider = EndpointProviderType(d.pop("provider"))

        in_registry = d.pop("in_registry", UNSET)

        _enterprise_status = d.pop("enterprise_status", UNSET)
        enterprise_status: Union[Unset, EnterpriseStatus]
        if isinstance(_enterprise_status, Unset):
            enterprise_status = UNSET
        else:
            enterprise_status = EnterpriseStatus(_enterprise_status)

        endpoint_price = cls(
            endpoint=endpoint,
            price=price,
            billable_unit=billable_unit,
            provider=provider,
            in_registry=in_registry,
            enterprise_status=enterprise_status,
        )

        endpoint_price.additional_properties = d
        return endpoint_price

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
