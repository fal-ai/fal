import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.runner_info_owner import RunnerInfoOwner
from ..models.runner_info_state import RunnerInfoState
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.runner_timing_info import RunnerTimingInfo


T = TypeVar("T", bound="RunnerInfo")


@_attrs_define
class RunnerInfo:
    """
    Attributes:
        runner_id (str):
        state (RunnerInfoState):
        machine_type (str):
        region (str):
        node_id (str):
        owner (RunnerInfoOwner):
        application (Union[Unset, str]):
        app_id (Union[Unset, str]):
        started_at (Union[Unset, datetime.datetime]):
        finished_at (Union[Unset, datetime.datetime]):
        created_at (Union[Unset, datetime.datetime]):
        cpu_usage_percent (Union[Unset, float]):
        memory_usage_bytes (Union[Unset, int]):
        vram_usage_ratio (Union[Unset, float]):
        expiration_countdown (Union[Unset, int]):  Default: 0.
        will_replace (Union[Unset, bool]):  Default: False.
        timing (Union[Unset, RunnerTimingInfo]): Timing information for runner state transitions.
        terminate_reason (Union[Unset, str]):
    """

    runner_id: str
    state: RunnerInfoState
    machine_type: str
    region: str
    node_id: str
    owner: RunnerInfoOwner
    application: Union[Unset, str] = UNSET
    app_id: Union[Unset, str] = UNSET
    started_at: Union[Unset, datetime.datetime] = UNSET
    finished_at: Union[Unset, datetime.datetime] = UNSET
    created_at: Union[Unset, datetime.datetime] = UNSET
    cpu_usage_percent: Union[Unset, float] = UNSET
    memory_usage_bytes: Union[Unset, int] = UNSET
    vram_usage_ratio: Union[Unset, float] = UNSET
    expiration_countdown: Union[Unset, int] = 0
    will_replace: Union[Unset, bool] = False
    timing: Union[Unset, "RunnerTimingInfo"] = UNSET
    terminate_reason: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        runner_id = self.runner_id

        state = self.state.value

        machine_type = self.machine_type

        region = self.region

        node_id = self.node_id

        owner = self.owner.value

        application = self.application

        app_id = self.app_id

        started_at: Union[Unset, str] = UNSET
        if not isinstance(self.started_at, Unset):
            started_at = self.started_at.isoformat()

        finished_at: Union[Unset, str] = UNSET
        if not isinstance(self.finished_at, Unset):
            finished_at = self.finished_at.isoformat()

        created_at: Union[Unset, str] = UNSET
        if not isinstance(self.created_at, Unset):
            created_at = self.created_at.isoformat()

        cpu_usage_percent = self.cpu_usage_percent

        memory_usage_bytes = self.memory_usage_bytes

        vram_usage_ratio = self.vram_usage_ratio

        expiration_countdown = self.expiration_countdown

        will_replace = self.will_replace

        timing: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.timing, Unset):
            timing = self.timing.to_dict()

        terminate_reason = self.terminate_reason

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "runner_id": runner_id,
                "state": state,
                "machine_type": machine_type,
                "region": region,
                "node_id": node_id,
                "owner": owner,
            }
        )
        if application is not UNSET:
            field_dict["application"] = application
        if app_id is not UNSET:
            field_dict["app_id"] = app_id
        if started_at is not UNSET:
            field_dict["started_at"] = started_at
        if finished_at is not UNSET:
            field_dict["finished_at"] = finished_at
        if created_at is not UNSET:
            field_dict["created_at"] = created_at
        if cpu_usage_percent is not UNSET:
            field_dict["cpu_usage_percent"] = cpu_usage_percent
        if memory_usage_bytes is not UNSET:
            field_dict["memory_usage_bytes"] = memory_usage_bytes
        if vram_usage_ratio is not UNSET:
            field_dict["vram_usage_ratio"] = vram_usage_ratio
        if expiration_countdown is not UNSET:
            field_dict["expiration_countdown"] = expiration_countdown
        if will_replace is not UNSET:
            field_dict["will_replace"] = will_replace
        if timing is not UNSET:
            field_dict["timing"] = timing
        if terminate_reason is not UNSET:
            field_dict["terminate_reason"] = terminate_reason

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.runner_timing_info import RunnerTimingInfo

        d = src_dict.copy()
        runner_id = d.pop("runner_id")

        state = RunnerInfoState(d.pop("state"))

        machine_type = d.pop("machine_type")

        region = d.pop("region")

        node_id = d.pop("node_id")

        owner = RunnerInfoOwner(d.pop("owner"))

        application = d.pop("application", UNSET)

        app_id = d.pop("app_id", UNSET)

        _started_at = d.pop("started_at", UNSET)
        started_at: Union[Unset, datetime.datetime]
        if isinstance(_started_at, Unset):
            started_at = UNSET
        else:
            started_at = isoparse(_started_at)

        _finished_at = d.pop("finished_at", UNSET)
        finished_at: Union[Unset, datetime.datetime]
        if isinstance(_finished_at, Unset):
            finished_at = UNSET
        else:
            finished_at = isoparse(_finished_at)

        _created_at = d.pop("created_at", UNSET)
        created_at: Union[Unset, datetime.datetime]
        if isinstance(_created_at, Unset):
            created_at = UNSET
        else:
            created_at = isoparse(_created_at)

        cpu_usage_percent = d.pop("cpu_usage_percent", UNSET)

        memory_usage_bytes = d.pop("memory_usage_bytes", UNSET)

        vram_usage_ratio = d.pop("vram_usage_ratio", UNSET)

        expiration_countdown = d.pop("expiration_countdown", UNSET)

        will_replace = d.pop("will_replace", UNSET)

        _timing = d.pop("timing", UNSET)
        timing: Union[Unset, RunnerTimingInfo]
        if isinstance(_timing, Unset):
            timing = UNSET
        else:
            timing = RunnerTimingInfo.from_dict(_timing)

        terminate_reason = d.pop("terminate_reason", UNSET)

        runner_info = cls(
            runner_id=runner_id,
            state=state,
            machine_type=machine_type,
            region=region,
            node_id=node_id,
            owner=owner,
            application=application,
            app_id=app_id,
            started_at=started_at,
            finished_at=finished_at,
            created_at=created_at,
            cpu_usage_percent=cpu_usage_percent,
            memory_usage_bytes=memory_usage_bytes,
            vram_usage_ratio=vram_usage_ratio,
            expiration_countdown=expiration_countdown,
            will_replace=will_replace,
            timing=timing,
            terminate_reason=terminate_reason,
        )

        runner_info.additional_properties = d
        return runner_info

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
