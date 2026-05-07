from typing import Any, TypeVar, Union
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.instance_status import InstanceStatus
from ..models.region import Region
from ..models.sector import Sector
from ..types import UNSET, Unset

T = TypeVar("T", bound="ComputeInstance")


@_attrs_define
class ComputeInstance:
    """
    Attributes:
        id (UUID):
        instance_type (str):
        region (Region):
        status (InstanceStatus):
        sector (Union[Unset, Sector]):
        ip (Union[Unset, str]):
        creator_user_nickname (Union[Unset, str]):
    """

    id: UUID
    instance_type: str
    region: Region
    status: InstanceStatus
    sector: Union[Unset, Sector] = UNSET
    ip: Union[Unset, str] = UNSET
    creator_user_nickname: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = str(self.id)

        instance_type = self.instance_type

        region = self.region.value

        status = self.status.value

        sector: Union[Unset, str] = UNSET
        if not isinstance(self.sector, Unset):
            sector = self.sector.value

        ip = self.ip

        creator_user_nickname = self.creator_user_nickname

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "instance_type": instance_type,
                "region": region,
                "status": status,
            }
        )
        if sector is not UNSET:
            field_dict["sector"] = sector
        if ip is not UNSET:
            field_dict["ip"] = ip
        if creator_user_nickname is not UNSET:
            field_dict["creator_user_nickname"] = creator_user_nickname

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        id = UUID(d.pop("id"))

        instance_type = d.pop("instance_type")

        region = Region(d.pop("region"))

        status = InstanceStatus(d.pop("status"))

        _sector = d.pop("sector", UNSET)
        sector: Union[Unset, Sector]
        if isinstance(_sector, Unset):
            sector = UNSET
        else:
            sector = Sector(_sector)

        ip = d.pop("ip", UNSET)

        creator_user_nickname = d.pop("creator_user_nickname", UNSET)

        compute_instance = cls(
            id=id,
            instance_type=instance_type,
            region=region,
            status=status,
            sector=sector,
            ip=ip,
            creator_user_nickname=creator_user_nickname,
        )

        compute_instance.additional_properties = d
        return compute_instance

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
