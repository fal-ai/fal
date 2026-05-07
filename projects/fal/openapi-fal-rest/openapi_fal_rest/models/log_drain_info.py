import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="LogDrainInfo")


@_attrs_define
class LogDrainInfo:
    """
    Attributes:
        drain_id (str):
        name (str):
        endpoint_url (str):
        sampling_rate (int):
        is_active (bool):
        created_at (datetime.datetime):
        consecutive_failures (int):
        updated_at (Union[Unset, datetime.datetime]):
        last_delivery_at (Union[Unset, datetime.datetime]):
        last_success_at (Union[Unset, datetime.datetime]):
    """

    drain_id: str
    name: str
    endpoint_url: str
    sampling_rate: int
    is_active: bool
    created_at: datetime.datetime
    consecutive_failures: int
    updated_at: Union[Unset, datetime.datetime] = UNSET
    last_delivery_at: Union[Unset, datetime.datetime] = UNSET
    last_success_at: Union[Unset, datetime.datetime] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        drain_id = self.drain_id

        name = self.name

        endpoint_url = self.endpoint_url

        sampling_rate = self.sampling_rate

        is_active = self.is_active

        created_at = self.created_at.isoformat()

        consecutive_failures = self.consecutive_failures

        updated_at: Union[Unset, str] = UNSET
        if not isinstance(self.updated_at, Unset):
            updated_at = self.updated_at.isoformat()

        last_delivery_at: Union[Unset, str] = UNSET
        if not isinstance(self.last_delivery_at, Unset):
            last_delivery_at = self.last_delivery_at.isoformat()

        last_success_at: Union[Unset, str] = UNSET
        if not isinstance(self.last_success_at, Unset):
            last_success_at = self.last_success_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "drain_id": drain_id,
                "name": name,
                "endpoint_url": endpoint_url,
                "sampling_rate": sampling_rate,
                "is_active": is_active,
                "created_at": created_at,
                "consecutive_failures": consecutive_failures,
            }
        )
        if updated_at is not UNSET:
            field_dict["updated_at"] = updated_at
        if last_delivery_at is not UNSET:
            field_dict["last_delivery_at"] = last_delivery_at
        if last_success_at is not UNSET:
            field_dict["last_success_at"] = last_success_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        drain_id = d.pop("drain_id")

        name = d.pop("name")

        endpoint_url = d.pop("endpoint_url")

        sampling_rate = d.pop("sampling_rate")

        is_active = d.pop("is_active")

        created_at = isoparse(d.pop("created_at"))

        consecutive_failures = d.pop("consecutive_failures")

        _updated_at = d.pop("updated_at", UNSET)
        updated_at: Union[Unset, datetime.datetime]
        if isinstance(_updated_at, Unset):
            updated_at = UNSET
        else:
            updated_at = isoparse(_updated_at)

        _last_delivery_at = d.pop("last_delivery_at", UNSET)
        last_delivery_at: Union[Unset, datetime.datetime]
        if isinstance(_last_delivery_at, Unset):
            last_delivery_at = UNSET
        else:
            last_delivery_at = isoparse(_last_delivery_at)

        _last_success_at = d.pop("last_success_at", UNSET)
        last_success_at: Union[Unset, datetime.datetime]
        if isinstance(_last_success_at, Unset):
            last_success_at = UNSET
        else:
            last_success_at = isoparse(_last_success_at)

        log_drain_info = cls(
            drain_id=drain_id,
            name=name,
            endpoint_url=endpoint_url,
            sampling_rate=sampling_rate,
            is_active=is_active,
            created_at=created_at,
            consecutive_failures=consecutive_failures,
            updated_at=updated_at,
            last_delivery_at=last_delivery_at,
            last_success_at=last_success_at,
        )

        log_drain_info.additional_properties = d
        return log_drain_info

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
