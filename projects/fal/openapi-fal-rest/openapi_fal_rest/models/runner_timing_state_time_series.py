from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="RunnerTimingStateTimeSeries")


@_attrs_define
class RunnerTimingStateTimeSeries:
    """
    Attributes:
        idle (list[list[float]]):
        busy (list[list[float]]):
        startup (list[list[float]]):
        schedule (list[list[float]]):
    """

    idle: list[list[float]]
    busy: list[list[float]]
    startup: list[list[float]]
    schedule: list[list[float]]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        idle = []
        for componentsschemas_time_series_item_data in self.idle:
            componentsschemas_time_series_item = componentsschemas_time_series_item_data

            idle.append(componentsschemas_time_series_item)

        busy = []
        for componentsschemas_time_series_item_data in self.busy:
            componentsschemas_time_series_item = componentsschemas_time_series_item_data

            busy.append(componentsschemas_time_series_item)

        startup = []
        for componentsschemas_time_series_item_data in self.startup:
            componentsschemas_time_series_item = componentsschemas_time_series_item_data

            startup.append(componentsschemas_time_series_item)

        schedule = []
        for componentsschemas_time_series_item_data in self.schedule:
            componentsschemas_time_series_item = componentsschemas_time_series_item_data

            schedule.append(componentsschemas_time_series_item)

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
        idle = []
        _idle = d.pop("idle")
        for componentsschemas_time_series_item_data in _idle:
            componentsschemas_time_series_item = cast(list[float], componentsschemas_time_series_item_data)

            idle.append(componentsschemas_time_series_item)

        busy = []
        _busy = d.pop("busy")
        for componentsschemas_time_series_item_data in _busy:
            componentsschemas_time_series_item = cast(list[float], componentsschemas_time_series_item_data)

            busy.append(componentsschemas_time_series_item)

        startup = []
        _startup = d.pop("startup")
        for componentsschemas_time_series_item_data in _startup:
            componentsschemas_time_series_item = cast(list[float], componentsschemas_time_series_item_data)

            startup.append(componentsschemas_time_series_item)

        schedule = []
        _schedule = d.pop("schedule")
        for componentsschemas_time_series_item_data in _schedule:
            componentsschemas_time_series_item = cast(list[float], componentsschemas_time_series_item_data)

            schedule.append(componentsschemas_time_series_item)

        runner_timing_state_time_series = cls(
            idle=idle,
            busy=busy,
            startup=startup,
            schedule=schedule,
        )

        runner_timing_state_time_series.additional_properties = d
        return runner_timing_state_time_series

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
