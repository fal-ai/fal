from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.provision_vercel_resource_request_metadata import ProvisionVercelResourceRequestMetadata


T = TypeVar("T", bound="ProvisionVercelResourceRequest")


@_attrs_define
class ProvisionVercelResourceRequest:
    """
    Attributes:
        billing_plan_id (Union[Unset, str]):
        product_id (Union[Unset, str]):
        name (Union[Unset, str]):
        metadata (Union[Unset, ProvisionVercelResourceRequestMetadata]):
    """

    billing_plan_id: Union[Unset, str] = UNSET
    product_id: Union[Unset, str] = UNSET
    name: Union[Unset, str] = UNSET
    metadata: Union[Unset, "ProvisionVercelResourceRequestMetadata"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        billing_plan_id = self.billing_plan_id

        product_id = self.product_id

        name = self.name

        metadata: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if billing_plan_id is not UNSET:
            field_dict["billingPlanId"] = billing_plan_id
        if product_id is not UNSET:
            field_dict["productId"] = product_id
        if name is not UNSET:
            field_dict["name"] = name
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.provision_vercel_resource_request_metadata import ProvisionVercelResourceRequestMetadata

        d = src_dict.copy()
        billing_plan_id = d.pop("billingPlanId", UNSET)

        product_id = d.pop("productId", UNSET)

        name = d.pop("name", UNSET)

        _metadata = d.pop("metadata", UNSET)
        metadata: Union[Unset, ProvisionVercelResourceRequestMetadata]
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = ProvisionVercelResourceRequestMetadata.from_dict(_metadata)

        provision_vercel_resource_request = cls(
            billing_plan_id=billing_plan_id,
            product_id=product_id,
            name=name,
            metadata=metadata,
        )

        provision_vercel_resource_request.additional_properties = d
        return provision_vercel_resource_request

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
