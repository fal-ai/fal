import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="RequestsByStatusCodePoint")


@_attrs_define
class RequestsByStatusCodePoint:
    """
    Attributes:
        datetime_ (datetime.datetime):
        status_1xx (Union[Unset, float]):  Default: 0.0.
        status_2xx (Union[Unset, float]):  Default: 0.0.
        status_3xx (Union[Unset, float]):  Default: 0.0.
        status_4xx (Union[Unset, float]):  Default: 0.0.
        status_5xx (Union[Unset, float]):  Default: 0.0.
    """

    datetime_: datetime.datetime
    status_1xx: Union[Unset, float] = 0.0
    status_2xx: Union[Unset, float] = 0.0
    status_3xx: Union[Unset, float] = 0.0
    status_4xx: Union[Unset, float] = 0.0
    status_5xx: Union[Unset, float] = 0.0
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        datetime_ = self.datetime_.isoformat()

        status_1xx = self.status_1xx

        status_2xx = self.status_2xx

        status_3xx = self.status_3xx

        status_4xx = self.status_4xx

        status_5xx = self.status_5xx

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "datetime": datetime_,
            }
        )
        if status_1xx is not UNSET:
            field_dict["status_1xx"] = status_1xx
        if status_2xx is not UNSET:
            field_dict["status_2xx"] = status_2xx
        if status_3xx is not UNSET:
            field_dict["status_3xx"] = status_3xx
        if status_4xx is not UNSET:
            field_dict["status_4xx"] = status_4xx
        if status_5xx is not UNSET:
            field_dict["status_5xx"] = status_5xx

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        datetime_ = isoparse(d.pop("datetime"))

        status_1xx = d.pop("status_1xx", UNSET)

        status_2xx = d.pop("status_2xx", UNSET)

        status_3xx = d.pop("status_3xx", UNSET)

        status_4xx = d.pop("status_4xx", UNSET)

        status_5xx = d.pop("status_5xx", UNSET)

        requests_by_status_code_point = cls(
            datetime_=datetime_,
            status_1xx=status_1xx,
            status_2xx=status_2xx,
            status_3xx=status_3xx,
            status_4xx=status_4xx,
            status_5xx=status_5xx,
        )

        requests_by_status_code_point.additional_properties = d
        return requests_by_status_code_point

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
