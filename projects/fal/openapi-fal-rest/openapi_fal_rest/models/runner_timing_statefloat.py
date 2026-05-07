from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="RunnerTimingStatefloat")


@_attrs_define
class RunnerTimingStatefloat:
    """
    Attributes:
        idle (float):
        busy (float):
        startup (float):
        schedule (float):
    """

    idle: float
    busy: float
    startup: float
    schedule: float
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        idle = self.idle

        busy = self.busy

        startup = self.startup

        schedule = self.schedule

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "idle": idle,
                "busy": busy,
                "startup": startup,
                "schedule": schedule,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        idle = d.pop("idle")

        busy = d.pop("busy")

        startup = d.pop("startup")

        schedule = d.pop("schedule")

        runner_timing_statefloat = cls(
            idle=idle,
            busy=busy,
            startup=startup,
            schedule=schedule,
        )

        runner_timing_statefloat.additional_properties = d
        return runner_timing_statefloat

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
