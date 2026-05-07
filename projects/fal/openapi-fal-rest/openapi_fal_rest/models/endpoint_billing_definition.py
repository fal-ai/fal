from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.endpoint_provider_type import EndpointProviderType
from ..models.enterprise_status import EnterpriseStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="EndpointBillingDefinition")


@_attrs_define
class EndpointBillingDefinition:
    """
    Attributes:
        endpoint (str):
        billing_unit (str):
        price (float):
        description (Union[Unset, str]):
        provider_type (Union[Unset, EndpointProviderType]):
        is_partner_api (Union[Unset, bool]):  Default: False.
        balance_check (Union[Unset, bool]):  Default: False.
        use_compute_seconds (Union[Unset, bool]):  Default: False.
        enterprise_status (Union[Unset, EnterpriseStatus]): Enterprise readiness status for endpoints.

            READY: Endpoint meets enterprise compliance standards (DPA, no training on data)
            PENDING: Endpoint is under review or does not yet meet enterprise standards
    """

    endpoint: str
    billing_unit: str
    price: float
    description: Union[Unset, str] = UNSET
    provider_type: Union[Unset, EndpointProviderType] = UNSET
    is_partner_api: Union[Unset, bool] = False
    balance_check: Union[Unset, bool] = False
    use_compute_seconds: Union[Unset, bool] = False
    enterprise_status: Union[Unset, EnterpriseStatus] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        endpoint = self.endpoint

        billing_unit = self.billing_unit

        price = self.price

        description = self.description

        provider_type: Union[Unset, str] = UNSET
        if not isinstance(self.provider_type, Unset):
            provider_type = self.provider_type.value

        is_partner_api = self.is_partner_api

        balance_check = self.balance_check

        use_compute_seconds = self.use_compute_seconds

        enterprise_status: Union[Unset, str] = UNSET
        if not isinstance(self.enterprise_status, Unset):
            enterprise_status = self.enterprise_status.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "endpoint": endpoint,
                "billing_unit": billing_unit,
                "price": price,
            }
        )
        if description is not UNSET:
            field_dict["description"] = description
        if provider_type is not UNSET:
            field_dict["provider_type"] = provider_type
        if is_partner_api is not UNSET:
            field_dict["is_partner_api"] = is_partner_api
        if balance_check is not UNSET:
            field_dict["balance_check"] = balance_check
        if use_compute_seconds is not UNSET:
            field_dict["use_compute_seconds"] = use_compute_seconds
        if enterprise_status is not UNSET:
            field_dict["enterprise_status"] = enterprise_status

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        endpoint = d.pop("endpoint")

        billing_unit = d.pop("billing_unit")

        price = d.pop("price")

        description = d.pop("description", UNSET)

        _provider_type = d.pop("provider_type", UNSET)
        provider_type: Union[Unset, EndpointProviderType]
        if isinstance(_provider_type, Unset):
            provider_type = UNSET
        else:
            provider_type = EndpointProviderType(_provider_type)

        is_partner_api = d.pop("is_partner_api", UNSET)

        balance_check = d.pop("balance_check", UNSET)

        use_compute_seconds = d.pop("use_compute_seconds", UNSET)

        _enterprise_status = d.pop("enterprise_status", UNSET)
        enterprise_status: Union[Unset, EnterpriseStatus]
        if isinstance(_enterprise_status, Unset):
            enterprise_status = UNSET
        else:
            enterprise_status = EnterpriseStatus(_enterprise_status)

        endpoint_billing_definition = cls(
            endpoint=endpoint,
            billing_unit=billing_unit,
            price=price,
            description=description,
            provider_type=provider_type,
            is_partner_api=is_partner_api,
            balance_check=balance_check,
            use_compute_seconds=use_compute_seconds,
            enterprise_status=enterprise_status,
        )

        endpoint_billing_definition.additional_properties = d
        return endpoint_billing_definition

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
