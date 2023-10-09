from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..models.status_health import StatusHealth
from ..types import UNSET, Unset

T = TypeVar("T", bound="Status")


@attr.s(auto_attribs=True)
class Status:
    """
    Attributes:
        health (Union[Unset, StatusHealth]):  Default: StatusHealth.HEALTHY.
        queue_size (Union[Unset, int]):
        median_processing_time (Union[Unset, float]):
    """

    health: Union[Unset, StatusHealth] = StatusHealth.HEALTHY
    queue_size: Union[Unset, int] = 0
    median_processing_time: Union[Unset, float] = 0.0
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        health: Union[Unset, str] = UNSET
        if not isinstance(self.health, Unset):
            health = self.health.value

        queue_size = self.queue_size
        median_processing_time = self.median_processing_time

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if health is not UNSET:
            field_dict["health"] = health
        if queue_size is not UNSET:
            field_dict["queue_size"] = queue_size
        if median_processing_time is not UNSET:
            field_dict["median_processing_time"] = median_processing_time

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        _health = d.pop("health", UNSET)
        health: Union[Unset, StatusHealth]
        if isinstance(_health, Unset):
            health = UNSET
        else:
            health = StatusHealth(_health)

        queue_size = d.pop("queue_size", UNSET)

        median_processing_time = d.pop("median_processing_time", UNSET)

        status = cls(
            health=health,
            queue_size=queue_size,
            median_processing_time=median_processing_time,
        )

        status.additional_properties = d
        return status

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
