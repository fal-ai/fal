import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.application import Application


T = TypeVar("T", bound="EndpointInfo")


@_attrs_define
class EndpointInfo:
    """
    Attributes:
        formatted (str):
        application (Application):
        last_used_at (datetime.datetime):
        mode (Union[Unset, str]):
    """

    formatted: str
    application: "Application"
    last_used_at: datetime.datetime
    mode: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        formatted = self.formatted

        application = self.application.to_dict()

        last_used_at = self.last_used_at.isoformat()

        mode = self.mode

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "formatted": formatted,
                "application": application,
                "last_used_at": last_used_at,
            }
        )
        if mode is not UNSET:
            field_dict["mode"] = mode

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.application import Application

        d = src_dict.copy()
        formatted = d.pop("formatted")

        application = Application.from_dict(d.pop("application"))

        last_used_at = isoparse(d.pop("last_used_at"))

        mode = d.pop("mode", UNSET)

        endpoint_info = cls(
            formatted=formatted,
            application=application,
            last_used_at=last_used_at,
            mode=mode,
        )

        endpoint_info.additional_properties = d
        return endpoint_info

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
