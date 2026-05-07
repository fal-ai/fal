import datetime
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

T = TypeVar("T", bound="UsedApp")


@_attrs_define
class UsedApp:
    """
    Attributes:
        caller_user_id (str):
        endpoint (str):
        last_used_at (datetime.datetime):
    """

    caller_user_id: str
    endpoint: str
    last_used_at: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        caller_user_id = self.caller_user_id

        endpoint = self.endpoint

        last_used_at = self.last_used_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "caller_user_id": caller_user_id,
                "endpoint": endpoint,
                "last_used_at": last_used_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        caller_user_id = d.pop("caller_user_id")

        endpoint = d.pop("endpoint")

        last_used_at = isoparse(d.pop("last_used_at"))

        used_app = cls(
            caller_user_id=caller_user_id,
            endpoint=endpoint,
            last_used_at=last_used_at,
        )

        used_app.additional_properties = d
        return used_app

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
