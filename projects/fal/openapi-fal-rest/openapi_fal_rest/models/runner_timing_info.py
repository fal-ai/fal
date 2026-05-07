import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="RunnerTimingInfo")


@_attrs_define
class RunnerTimingInfo:
    """Timing information for runner state transitions.

    Attributes:
        pending_at (Union[Unset, datetime.datetime]):
        docker_pull_at (Union[Unset, datetime.datetime]):
        setup_at (Union[Unset, datetime.datetime]):
        started_at (Union[Unset, datetime.datetime]):
        pending_duration (Union[Unset, float]):
        docker_pull_duration (Union[Unset, float]):
        setup_duration (Union[Unset, float]):
        cold_start_duration (Union[Unset, float]):
    """

    pending_at: Union[Unset, datetime.datetime] = UNSET
    docker_pull_at: Union[Unset, datetime.datetime] = UNSET
    setup_at: Union[Unset, datetime.datetime] = UNSET
    started_at: Union[Unset, datetime.datetime] = UNSET
    pending_duration: Union[Unset, float] = UNSET
    docker_pull_duration: Union[Unset, float] = UNSET
    setup_duration: Union[Unset, float] = UNSET
    cold_start_duration: Union[Unset, float] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        pending_at: Union[Unset, str] = UNSET
        if not isinstance(self.pending_at, Unset):
            pending_at = self.pending_at.isoformat()

        docker_pull_at: Union[Unset, str] = UNSET
        if not isinstance(self.docker_pull_at, Unset):
            docker_pull_at = self.docker_pull_at.isoformat()

        setup_at: Union[Unset, str] = UNSET
        if not isinstance(self.setup_at, Unset):
            setup_at = self.setup_at.isoformat()

        started_at: Union[Unset, str] = UNSET
        if not isinstance(self.started_at, Unset):
            started_at = self.started_at.isoformat()

        pending_duration = self.pending_duration

        docker_pull_duration = self.docker_pull_duration

        setup_duration = self.setup_duration

        cold_start_duration = self.cold_start_duration

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if pending_at is not UNSET:
            field_dict["pending_at"] = pending_at
        if docker_pull_at is not UNSET:
            field_dict["docker_pull_at"] = docker_pull_at
        if setup_at is not UNSET:
            field_dict["setup_at"] = setup_at
        if started_at is not UNSET:
            field_dict["started_at"] = started_at
        if pending_duration is not UNSET:
            field_dict["pending_duration"] = pending_duration
        if docker_pull_duration is not UNSET:
            field_dict["docker_pull_duration"] = docker_pull_duration
        if setup_duration is not UNSET:
            field_dict["setup_duration"] = setup_duration
        if cold_start_duration is not UNSET:
            field_dict["cold_start_duration"] = cold_start_duration

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        _pending_at = d.pop("pending_at", UNSET)
        pending_at: Union[Unset, datetime.datetime]
        if isinstance(_pending_at, Unset):
            pending_at = UNSET
        else:
            pending_at = isoparse(_pending_at)

        _docker_pull_at = d.pop("docker_pull_at", UNSET)
        docker_pull_at: Union[Unset, datetime.datetime]
        if isinstance(_docker_pull_at, Unset):
            docker_pull_at = UNSET
        else:
            docker_pull_at = isoparse(_docker_pull_at)

        _setup_at = d.pop("setup_at", UNSET)
        setup_at: Union[Unset, datetime.datetime]
        if isinstance(_setup_at, Unset):
            setup_at = UNSET
        else:
            setup_at = isoparse(_setup_at)

        _started_at = d.pop("started_at", UNSET)
        started_at: Union[Unset, datetime.datetime]
        if isinstance(_started_at, Unset):
            started_at = UNSET
        else:
            started_at = isoparse(_started_at)

        pending_duration = d.pop("pending_duration", UNSET)

        docker_pull_duration = d.pop("docker_pull_duration", UNSET)

        setup_duration = d.pop("setup_duration", UNSET)

        cold_start_duration = d.pop("cold_start_duration", UNSET)

        runner_timing_info = cls(
            pending_at=pending_at,
            docker_pull_at=docker_pull_at,
            setup_at=setup_at,
            started_at=started_at,
            pending_duration=pending_duration,
            docker_pull_duration=docker_pull_duration,
            setup_duration=setup_duration,
            cold_start_duration=cold_start_duration,
        )

        runner_timing_info.additional_properties = d
        return runner_timing_info

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
