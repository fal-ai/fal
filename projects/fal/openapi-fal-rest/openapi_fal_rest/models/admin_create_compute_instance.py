import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.region import Region
from ..models.sector import Sector
from ..types import UNSET, Unset

T = TypeVar("T", bound="AdminCreateComputeInstance")


@_attrs_define
class AdminCreateComputeInstance:
    """
    Attributes:
        user_id (str):
        instance_type (str):
        ip_address (str):
        provider (str):
        execution_start_time (Union[Unset, datetime.datetime]):
        region (Union[Unset, Region]):
        sector (Union[Unset, Sector]):
        provider_id (Union[Unset, str]):
        ssh_key (Union[Unset, str]):
    """

    user_id: str
    instance_type: str
    ip_address: str
    provider: str
    execution_start_time: Union[Unset, datetime.datetime] = UNSET
    region: Union[Unset, Region] = UNSET
    sector: Union[Unset, Sector] = UNSET
    provider_id: Union[Unset, str] = UNSET
    ssh_key: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        instance_type = self.instance_type

        ip_address = self.ip_address

        provider = self.provider

        execution_start_time: Union[Unset, str] = UNSET
        if not isinstance(self.execution_start_time, Unset):
            execution_start_time = self.execution_start_time.isoformat()

        region: Union[Unset, str] = UNSET
        if not isinstance(self.region, Unset):
            region = self.region.value

        sector: Union[Unset, str] = UNSET
        if not isinstance(self.sector, Unset):
            sector = self.sector.value

        provider_id = self.provider_id

        ssh_key = self.ssh_key

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "instance_type": instance_type,
                "ip_address": ip_address,
                "provider": provider,
            }
        )
        if execution_start_time is not UNSET:
            field_dict["execution_start_time"] = execution_start_time
        if region is not UNSET:
            field_dict["region"] = region
        if sector is not UNSET:
            field_dict["sector"] = sector
        if provider_id is not UNSET:
            field_dict["provider_id"] = provider_id
        if ssh_key is not UNSET:
            field_dict["ssh_key"] = ssh_key

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        instance_type = d.pop("instance_type")

        ip_address = d.pop("ip_address")

        provider = d.pop("provider")

        _execution_start_time = d.pop("execution_start_time", UNSET)
        execution_start_time: Union[Unset, datetime.datetime]
        if isinstance(_execution_start_time, Unset):
            execution_start_time = UNSET
        else:
            execution_start_time = isoparse(_execution_start_time)

        _region = d.pop("region", UNSET)
        region: Union[Unset, Region]
        if isinstance(_region, Unset):
            region = UNSET
        else:
            region = Region(_region)

        _sector = d.pop("sector", UNSET)
        sector: Union[Unset, Sector]
        if isinstance(_sector, Unset):
            sector = UNSET
        else:
            sector = Sector(_sector)

        provider_id = d.pop("provider_id", UNSET)

        ssh_key = d.pop("ssh_key", UNSET)

        admin_create_compute_instance = cls(
            user_id=user_id,
            instance_type=instance_type,
            ip_address=ip_address,
            provider=provider,
            execution_start_time=execution_start_time,
            region=region,
            sector=sector,
            provider_id=provider_id,
            ssh_key=ssh_key,
        )

        admin_create_compute_instance.additional_properties = d
        return admin_create_compute_instance

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
