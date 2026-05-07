from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.provision_vercel_resource_response_status import ProvisionVercelResourceResponseStatus

if TYPE_CHECKING:
    from ..models.provision_vercel_resource_response_metadata import ProvisionVercelResourceResponseMetadata
    from ..models.resource_secret import ResourceSecret
    from ..models.vercel_installation_plan import VercelInstallationPlan


T = TypeVar("T", bound="ProvisionVercelResourceResponse")


@_attrs_define
class ProvisionVercelResourceResponse:
    """
    Attributes:
        id (str):
        name (str):
        metadata (ProvisionVercelResourceResponseMetadata):
        product_id (str):
        secrets (list['ResourceSecret']):
        status (ProvisionVercelResourceResponseStatus):
        billing_plan (VercelInstallationPlan):
    """

    id: str
    name: str
    metadata: "ProvisionVercelResourceResponseMetadata"
    product_id: str
    secrets: list["ResourceSecret"]
    status: ProvisionVercelResourceResponseStatus
    billing_plan: "VercelInstallationPlan"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        name = self.name

        metadata = self.metadata.to_dict()

        product_id = self.product_id

        secrets = []
        for secrets_item_data in self.secrets:
            secrets_item = secrets_item_data.to_dict()
            secrets.append(secrets_item)

        status = self.status.value

        billing_plan = self.billing_plan.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "name": name,
                "metadata": metadata,
                "productId": product_id,
                "secrets": secrets,
                "status": status,
                "billingPlan": billing_plan,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.provision_vercel_resource_response_metadata import ProvisionVercelResourceResponseMetadata
        from ..models.resource_secret import ResourceSecret
        from ..models.vercel_installation_plan import VercelInstallationPlan

        d = src_dict.copy()
        id = d.pop("id")

        name = d.pop("name")

        metadata = ProvisionVercelResourceResponseMetadata.from_dict(d.pop("metadata"))

        product_id = d.pop("productId")

        secrets = []
        _secrets = d.pop("secrets")
        for secrets_item_data in _secrets:
            secrets_item = ResourceSecret.from_dict(secrets_item_data)

            secrets.append(secrets_item)

        status = ProvisionVercelResourceResponseStatus(d.pop("status"))

        billing_plan = VercelInstallationPlan.from_dict(d.pop("billingPlan"))

        provision_vercel_resource_response = cls(
            id=id,
            name=name,
            metadata=metadata,
            product_id=product_id,
            secrets=secrets,
            status=status,
            billing_plan=billing_plan,
        )

        provision_vercel_resource_response.additional_properties = d
        return provision_vercel_resource_response

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
