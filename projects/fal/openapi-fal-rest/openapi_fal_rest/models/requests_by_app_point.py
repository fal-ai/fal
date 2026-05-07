import datetime
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

if TYPE_CHECKING:
    from ..models.requests_by_app_point_apps import RequestsByAppPointApps


T = TypeVar("T", bound="RequestsByAppPoint")


@_attrs_define
class RequestsByAppPoint:
    """
    Attributes:
        datetime_ (datetime.datetime):
        apps (RequestsByAppPointApps):
    """

    datetime_: datetime.datetime
    apps: "RequestsByAppPointApps"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        datetime_ = self.datetime_.isoformat()

        apps = self.apps.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "datetime": datetime_,
                "apps": apps,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.requests_by_app_point_apps import RequestsByAppPointApps

        d = src_dict.copy()
        datetime_ = isoparse(d.pop("datetime"))

        apps = RequestsByAppPointApps.from_dict(d.pop("apps"))

        requests_by_app_point = cls(
            datetime_=datetime_,
            apps=apps,
        )

        requests_by_app_point.additional_properties = d
        return requests_by_app_point

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
