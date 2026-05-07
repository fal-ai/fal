import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="JobStats")


@_attrs_define
class JobStats:
    """
    Attributes:
        job_id (str):
        start_time (datetime.datetime):
        setup_duration (float):
        idle_duration (float):
        request_duration (float):
        end_time (Union[Unset, datetime.datetime]):
        setup_ready_time (Union[Unset, datetime.datetime]):
    """

    job_id: str
    start_time: datetime.datetime
    setup_duration: float
    idle_duration: float
    request_duration: float
    end_time: Union[Unset, datetime.datetime] = UNSET
    setup_ready_time: Union[Unset, datetime.datetime] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        job_id = self.job_id

        start_time = self.start_time.isoformat()

        setup_duration = self.setup_duration

        idle_duration = self.idle_duration

        request_duration = self.request_duration

        end_time: Union[Unset, str] = UNSET
        if not isinstance(self.end_time, Unset):
            end_time = self.end_time.isoformat()

        setup_ready_time: Union[Unset, str] = UNSET
        if not isinstance(self.setup_ready_time, Unset):
            setup_ready_time = self.setup_ready_time.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "job_id": job_id,
                "start_time": start_time,
                "setup_duration": setup_duration,
                "idle_duration": idle_duration,
                "request_duration": request_duration,
            }
        )
        if end_time is not UNSET:
            field_dict["end_time"] = end_time
        if setup_ready_time is not UNSET:
            field_dict["setup_ready_time"] = setup_ready_time

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        job_id = d.pop("job_id")

        start_time = isoparse(d.pop("start_time"))

        setup_duration = d.pop("setup_duration")

        idle_duration = d.pop("idle_duration")

        request_duration = d.pop("request_duration")

        _end_time = d.pop("end_time", UNSET)
        end_time: Union[Unset, datetime.datetime]
        if isinstance(_end_time, Unset):
            end_time = UNSET
        else:
            end_time = isoparse(_end_time)

        _setup_ready_time = d.pop("setup_ready_time", UNSET)
        setup_ready_time: Union[Unset, datetime.datetime]
        if isinstance(_setup_ready_time, Unset):
            setup_ready_time = UNSET
        else:
            setup_ready_time = isoparse(_setup_ready_time)

        job_stats = cls(
            job_id=job_id,
            start_time=start_time,
            setup_duration=setup_duration,
            idle_duration=idle_duration,
            request_duration=request_duration,
            end_time=end_time,
            setup_ready_time=setup_ready_time,
        )

        job_stats.additional_properties = d
        return job_stats

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
