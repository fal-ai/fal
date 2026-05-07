from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.provision_vercel_resource_response import ProvisionVercelResourceResponse


T = TypeVar("T", bound="ListResourcesResponse")


@_attrs_define
class ListResourcesResponse:
    """
    Attributes:
        resources (list['ProvisionVercelResourceResponse']):
    """

    resources: list["ProvisionVercelResourceResponse"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        resources = []
        for resources_item_data in self.resources:
            resources_item = resources_item_data.to_dict()
            resources.append(resources_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "resources": resources,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.provision_vercel_resource_response import ProvisionVercelResourceResponse

        d = src_dict.copy()
        resources = []
        _resources = d.pop("resources")
        for resources_item_data in _resources:
            resources_item = ProvisionVercelResourceResponse.from_dict(resources_item_data)

            resources.append(resources_item)

        list_resources_response = cls(
            resources=resources,
        )

        list_resources_response.additional_properties = d
        return list_resources_response

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
