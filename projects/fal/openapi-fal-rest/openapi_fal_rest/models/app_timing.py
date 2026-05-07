from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.runner_timing_state_time_series import RunnerTimingStateTimeSeries
    from ..models.runner_timing_statefloat import RunnerTimingStatefloat


T = TypeVar("T", bound="AppTiming")


@_attrs_define
class AppTiming:
    """
    Attributes:
        range_ (RunnerTimingStateTimeSeries):
        instant (RunnerTimingStatefloat):
    """

    range_: "RunnerTimingStateTimeSeries"
    instant: "RunnerTimingStatefloat"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        range_ = self.range_.to_dict()

        instant = self.instant.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "range": range_,
                "instant": instant,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.runner_timing_state_time_series import RunnerTimingStateTimeSeries
        from ..models.runner_timing_statefloat import RunnerTimingStatefloat

        d = src_dict.copy()
        range_ = RunnerTimingStateTimeSeries.from_dict(d.pop("range"))

        instant = RunnerTimingStatefloat.from_dict(d.pop("instant"))

        app_timing = cls(
            range_=range_,
            instant=instant,
        )

        app_timing.additional_properties = d
        return app_timing

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
