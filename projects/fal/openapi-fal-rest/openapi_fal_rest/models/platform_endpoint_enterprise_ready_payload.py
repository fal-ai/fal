from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="PlatformEndpointEnterpriseReadyPayload")


@_attrs_define
class PlatformEndpointEnterpriseReadyPayload:
    """
    Attributes:
        endpoint (Union[Unset, str]):
        model_name (Union[Unset, str]):
        description (Union[Unset, str]):
        ready_at (Union[Unset, str]):
        org_user_full_name (Union[Unset, str]):
        playground_url (Union[Unset, str]):
    """

    endpoint: Union[Unset, str] = UNSET
    model_name: Union[Unset, str] = UNSET
    description: Union[Unset, str] = UNSET
    ready_at: Union[Unset, str] = UNSET
    org_user_full_name: Union[Unset, str] = UNSET
    playground_url: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        endpoint = self.endpoint

        model_name = self.model_name

        description = self.description

        ready_at = self.ready_at

        org_user_full_name = self.org_user_full_name

        playground_url = self.playground_url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if endpoint is not UNSET:
            field_dict["endpoint"] = endpoint
        if model_name is not UNSET:
            field_dict["model_name"] = model_name
        if description is not UNSET:
            field_dict["description"] = description
        if ready_at is not UNSET:
            field_dict["ready_at"] = ready_at
        if org_user_full_name is not UNSET:
            field_dict["org_user_full_name"] = org_user_full_name
        if playground_url is not UNSET:
            field_dict["playground_url"] = playground_url

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        endpoint = d.pop("endpoint", UNSET)

        model_name = d.pop("model_name", UNSET)

        description = d.pop("description", UNSET)

        ready_at = d.pop("ready_at", UNSET)

        org_user_full_name = d.pop("org_user_full_name", UNSET)

        playground_url = d.pop("playground_url", UNSET)

        platform_endpoint_enterprise_ready_payload = cls(
            endpoint=endpoint,
            model_name=model_name,
            description=description,
            ready_at=ready_at,
            org_user_full_name=org_user_full_name,
            playground_url=playground_url,
        )

        platform_endpoint_enterprise_ready_payload.additional_properties = d
        return platform_endpoint_enterprise_ready_payload

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
