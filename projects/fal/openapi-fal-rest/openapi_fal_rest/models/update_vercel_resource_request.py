from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.update_vercel_resource_request_status import UpdateVercelResourceRequestStatus

if TYPE_CHECKING:
    from ..models.update_vercel_resource_request_metadata import UpdateVercelResourceRequestMetadata


T = TypeVar("T", bound="UpdateVercelResourceRequest")


@_attrs_define
class UpdateVercelResourceRequest:
    """
    Attributes:
        name (str):
        metadata (UpdateVercelResourceRequestMetadata):
        billing_plan_id (str):
        status (UpdateVercelResourceRequestStatus):
    """

    name: str
    metadata: "UpdateVercelResourceRequestMetadata"
    billing_plan_id: str
    status: UpdateVercelResourceRequestStatus
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        metadata = self.metadata.to_dict()

        billing_plan_id = self.billing_plan_id

        status = self.status.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "metadata": metadata,
                "billingPlanId": billing_plan_id,
                "status": status,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.update_vercel_resource_request_metadata import UpdateVercelResourceRequestMetadata

        d = src_dict.copy()
        name = d.pop("name")

        metadata = UpdateVercelResourceRequestMetadata.from_dict(d.pop("metadata"))

        billing_plan_id = d.pop("billingPlanId")

        status = UpdateVercelResourceRequestStatus(d.pop("status"))

        update_vercel_resource_request = cls(
            name=name,
            metadata=metadata,
            billing_plan_id=billing_plan_id,
            status=status,
        )

        update_vercel_resource_request.additional_properties = d
        return update_vercel_resource_request

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
