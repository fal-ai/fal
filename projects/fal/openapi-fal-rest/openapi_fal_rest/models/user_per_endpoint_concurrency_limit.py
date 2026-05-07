from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="UserPerEndpointConcurrencyLimit")


@_attrs_define
class UserPerEndpointConcurrencyLimit:
    """
    Attributes:
        endpoints (list[str]):
        limit (Union[Unset, int]):
    """

    endpoints: list[str]
    limit: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        endpoints = self.endpoints

        limit = self.limit

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "endpoints": endpoints,
            }
        )
        if limit is not UNSET:
            field_dict["limit"] = limit

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        endpoints = cast(list[str], d.pop("endpoints"))

        limit = d.pop("limit", UNSET)

        user_per_endpoint_concurrency_limit = cls(
            endpoints=endpoints,
            limit=limit,
        )

        user_per_endpoint_concurrency_limit.additional_properties = d
        return user_per_endpoint_concurrency_limit

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
