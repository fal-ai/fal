from typing import Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="CostsRequest")


@_attrs_define
class CostsRequest:
    """
    Attributes:
        request_ids (list[UUID]):
    """

    request_ids: list[UUID]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        request_ids = []
        for request_ids_item_data in self.request_ids:
            request_ids_item = str(request_ids_item_data)
            request_ids.append(request_ids_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "requestIds": request_ids,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        request_ids = []
        _request_ids = d.pop("requestIds")
        for request_ids_item_data in _request_ids:
            request_ids_item = UUID(request_ids_item_data)

            request_ids.append(request_ids_item)

        costs_request = cls(
            request_ids=request_ids,
        )

        costs_request.additional_properties = d
        return costs_request

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
