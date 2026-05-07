from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.instance_type import InstanceType
from ..models.region import Region
from ..models.sector import Sector
from ..types import UNSET, Unset

T = TypeVar("T", bound="CreateComputeInstance")


@_attrs_define
class CreateComputeInstance:
    """
    Attributes:
        instance_type (InstanceType):
        ssh_key (str):
        region (Union[Unset, Region]):
        sector (Union[Unset, Sector]):
    """

    instance_type: InstanceType
    ssh_key: str
    region: Union[Unset, Region] = UNSET
    sector: Union[Unset, Sector] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        instance_type = self.instance_type.value

        ssh_key = self.ssh_key

        region: Union[Unset, str] = UNSET
        if not isinstance(self.region, Unset):
            region = self.region.value

        sector: Union[Unset, str] = UNSET
        if not isinstance(self.sector, Unset):
            sector = self.sector.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "instance_type": instance_type,
                "ssh_key": ssh_key,
            }
        )
        if region is not UNSET:
            field_dict["region"] = region
        if sector is not UNSET:
            field_dict["sector"] = sector

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        instance_type = InstanceType(d.pop("instance_type"))

        ssh_key = d.pop("ssh_key")

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

        create_compute_instance = cls(
            instance_type=instance_type,
            ssh_key=ssh_key,
            region=region,
            sector=sector,
        )

        create_compute_instance.additional_properties = d
        return create_compute_instance

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
