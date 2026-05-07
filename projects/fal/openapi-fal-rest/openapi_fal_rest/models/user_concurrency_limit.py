from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.user_concurrency_limit_concurrent_requests_limit_per_endpoint import (
        UserConcurrencyLimitConcurrentRequestsLimitPerEndpoint,
    )


T = TypeVar("T", bound="UserConcurrencyLimit")


@_attrs_define
class UserConcurrencyLimit:
    """
    Attributes:
        user_id (str):
        concurrent_requests_limit (Union[Unset, int]):
        concurrent_requests_limit_per_endpoint (Union[Unset, UserConcurrencyLimitConcurrentRequestsLimitPerEndpoint]):
    """

    user_id: str
    concurrent_requests_limit: Union[Unset, int] = UNSET
    concurrent_requests_limit_per_endpoint: Union[Unset, "UserConcurrencyLimitConcurrentRequestsLimitPerEndpoint"] = (
        UNSET
    )
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        concurrent_requests_limit = self.concurrent_requests_limit

        concurrent_requests_limit_per_endpoint: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.concurrent_requests_limit_per_endpoint, Unset):
            concurrent_requests_limit_per_endpoint = self.concurrent_requests_limit_per_endpoint.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
            }
        )
        if concurrent_requests_limit is not UNSET:
            field_dict["concurrent_requests_limit"] = concurrent_requests_limit
        if concurrent_requests_limit_per_endpoint is not UNSET:
            field_dict["concurrent_requests_limit_per_endpoint"] = concurrent_requests_limit_per_endpoint

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.user_concurrency_limit_concurrent_requests_limit_per_endpoint import (
            UserConcurrencyLimitConcurrentRequestsLimitPerEndpoint,
        )

        d = src_dict.copy()
        user_id = d.pop("user_id")

        concurrent_requests_limit = d.pop("concurrent_requests_limit", UNSET)

        _concurrent_requests_limit_per_endpoint = d.pop("concurrent_requests_limit_per_endpoint", UNSET)
        concurrent_requests_limit_per_endpoint: Union[Unset, UserConcurrencyLimitConcurrentRequestsLimitPerEndpoint]
        if isinstance(_concurrent_requests_limit_per_endpoint, Unset):
            concurrent_requests_limit_per_endpoint = UNSET
        else:
            concurrent_requests_limit_per_endpoint = UserConcurrencyLimitConcurrentRequestsLimitPerEndpoint.from_dict(
                _concurrent_requests_limit_per_endpoint
            )

        user_concurrency_limit = cls(
            user_id=user_id,
            concurrent_requests_limit=concurrent_requests_limit,
            concurrent_requests_limit_per_endpoint=concurrent_requests_limit_per_endpoint,
        )

        user_concurrency_limit.additional_properties = d
        return user_concurrency_limit

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
