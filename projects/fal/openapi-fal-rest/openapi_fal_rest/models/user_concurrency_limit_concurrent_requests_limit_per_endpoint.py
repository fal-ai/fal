from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.user_per_endpoint_concurrency_limit import UserPerEndpointConcurrencyLimit


T = TypeVar("T", bound="UserConcurrencyLimitConcurrentRequestsLimitPerEndpoint")


@_attrs_define
class UserConcurrencyLimitConcurrentRequestsLimitPerEndpoint:
    """ """

    additional_properties: dict[str, "UserPerEndpointConcurrencyLimit"] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.user_per_endpoint_concurrency_limit import UserPerEndpointConcurrencyLimit

        d = src_dict.copy()
        user_concurrency_limit_concurrent_requests_limit_per_endpoint = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = UserPerEndpointConcurrencyLimit.from_dict(prop_dict)

            additional_properties[prop_name] = additional_property

        user_concurrency_limit_concurrent_requests_limit_per_endpoint.additional_properties = additional_properties
        return user_concurrency_limit_concurrent_requests_limit_per_endpoint

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> "UserPerEndpointConcurrencyLimit":
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: "UserPerEndpointConcurrencyLimit") -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
