from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="RunnerStateSeries")


@_attrs_define
class RunnerStateSeries:
    """
    Attributes:
        running (list[list[float]]):
        draining (list[list[float]]):
        pending (list[list[float]]):
        idle (list[list[float]]):
    """

    running: list[list[float]]
    draining: list[list[float]]
    pending: list[list[float]]
    idle: list[list[float]]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        running = []
        for componentsschemas_time_series_item_data in self.running:
            componentsschemas_time_series_item = componentsschemas_time_series_item_data

            running.append(componentsschemas_time_series_item)

        draining = []
        for componentsschemas_time_series_item_data in self.draining:
            componentsschemas_time_series_item = componentsschemas_time_series_item_data

            draining.append(componentsschemas_time_series_item)

        pending = []
        for componentsschemas_time_series_item_data in self.pending:
            componentsschemas_time_series_item = componentsschemas_time_series_item_data

            pending.append(componentsschemas_time_series_item)

        idle = []
        for componentsschemas_time_series_item_data in self.idle:
            componentsschemas_time_series_item = componentsschemas_time_series_item_data

            idle.append(componentsschemas_time_series_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "running": running,
                "draining": draining,
                "pending": pending,
                "idle": idle,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        running = []
        _running = d.pop("running")
        for componentsschemas_time_series_item_data in _running:
            componentsschemas_time_series_item = cast(list[float], componentsschemas_time_series_item_data)

            running.append(componentsschemas_time_series_item)

        draining = []
        _draining = d.pop("draining")
        for componentsschemas_time_series_item_data in _draining:
            componentsschemas_time_series_item = cast(list[float], componentsschemas_time_series_item_data)

            draining.append(componentsschemas_time_series_item)

        pending = []
        _pending = d.pop("pending")
        for componentsschemas_time_series_item_data in _pending:
            componentsschemas_time_series_item = cast(list[float], componentsschemas_time_series_item_data)

            pending.append(componentsschemas_time_series_item)

        idle = []
        _idle = d.pop("idle")
        for componentsschemas_time_series_item_data in _idle:
            componentsschemas_time_series_item = cast(list[float], componentsschemas_time_series_item_data)

            idle.append(componentsschemas_time_series_item)

        runner_state_series = cls(
            running=running,
            draining=draining,
            pending=pending,
            idle=idle,
        )

        runner_state_series.additional_properties = d
        return runner_state_series

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
