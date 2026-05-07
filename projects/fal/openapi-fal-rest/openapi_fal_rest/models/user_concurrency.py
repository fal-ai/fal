from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="UserConcurrency")


@_attrs_define
class UserConcurrency:
    """
    Attributes:
        user_id (str):
        concurrent_requests (int):
        concurrent_requests_limit (Union[Unset, int]):
    """

    user_id: str
    concurrent_requests: int
    concurrent_requests_limit: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        concurrent_requests = self.concurrent_requests

        concurrent_requests_limit = self.concurrent_requests_limit

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "concurrent_requests": concurrent_requests,
            }
        )
        if concurrent_requests_limit is not UNSET:
            field_dict["concurrent_requests_limit"] = concurrent_requests_limit

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        concurrent_requests = d.pop("concurrent_requests")

        concurrent_requests_limit = d.pop("concurrent_requests_limit", UNSET)

        user_concurrency = cls(
            user_id=user_id,
            concurrent_requests=concurrent_requests,
            concurrent_requests_limit=concurrent_requests_limit,
        )

        user_concurrency.additional_properties = d
        return user_concurrency

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
