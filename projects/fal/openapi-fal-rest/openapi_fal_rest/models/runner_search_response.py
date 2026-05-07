import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.runner_search_response_owner import RunnerSearchResponseOwner
from ..models.runner_search_response_state import RunnerSearchResponseState
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.cpu_stats_by_time import CpuStatsByTime
    from ..models.memory_stats_by_time import MemoryStatsByTime
    from ..models.runner_timing_info import RunnerTimingInfo
    from ..models.vram_stats_by_time import VramStatsByTime


T = TypeVar("T", bound="RunnerSearchResponse")


@_attrs_define
class RunnerSearchResponse:
    """Extended runner details for admin search, includes user and app info.

    Attributes:
        runner_id (str):
        state (RunnerSearchResponseState):
        machine_type (str):
        region (str):
        node_id (str):
        user_id (str):
        application (Union[Unset, str]):
        application_id (Union[Unset, str]):
        node_name (Union[Unset, str]):
        node_data_center (Union[Unset, str]):
        started_at (Union[Unset, datetime.datetime]):
        finished_at (Union[Unset, datetime.datetime]):
        owner (Union[Unset, RunnerSearchResponseOwner]):
        user_nickname (Union[Unset, str]):
        job_source (Union[Unset, str]):
        status (Union[Unset, str]):
        created_at (Union[Unset, datetime.datetime]):
        cpu_usage_history (Union[Unset, list['CpuStatsByTime']]):
        memory_usage_history (Union[Unset, list['MemoryStatsByTime']]):
        vram_usage_history (Union[Unset, list['VramStatsByTime']]):
        timing (Union[Unset, RunnerTimingInfo]): Timing information for runner state transitions.
        terminate_reason (Union[Unset, str]):
    """

    runner_id: str
    state: RunnerSearchResponseState
    machine_type: str
    region: str
    node_id: str
    user_id: str
    application: Union[Unset, str] = UNSET
    application_id: Union[Unset, str] = UNSET
    node_name: Union[Unset, str] = UNSET
    node_data_center: Union[Unset, str] = UNSET
    started_at: Union[Unset, datetime.datetime] = UNSET
    finished_at: Union[Unset, datetime.datetime] = UNSET
    owner: Union[Unset, RunnerSearchResponseOwner] = UNSET
    user_nickname: Union[Unset, str] = UNSET
    job_source: Union[Unset, str] = UNSET
    status: Union[Unset, str] = UNSET
    created_at: Union[Unset, datetime.datetime] = UNSET
    cpu_usage_history: Union[Unset, list["CpuStatsByTime"]] = UNSET
    memory_usage_history: Union[Unset, list["MemoryStatsByTime"]] = UNSET
    vram_usage_history: Union[Unset, list["VramStatsByTime"]] = UNSET
    timing: Union[Unset, "RunnerTimingInfo"] = UNSET
    terminate_reason: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        runner_id = self.runner_id

        state = self.state.value

        machine_type = self.machine_type

        region = self.region

        node_id = self.node_id

        user_id = self.user_id

        application = self.application

        application_id = self.application_id

        node_name = self.node_name

        node_data_center = self.node_data_center

        started_at: Union[Unset, str] = UNSET
        if not isinstance(self.started_at, Unset):
            started_at = self.started_at.isoformat()

        finished_at: Union[Unset, str] = UNSET
        if not isinstance(self.finished_at, Unset):
            finished_at = self.finished_at.isoformat()

        owner: Union[Unset, str] = UNSET
        if not isinstance(self.owner, Unset):
            owner = self.owner.value

        user_nickname = self.user_nickname

        job_source = self.job_source

        status = self.status

        created_at: Union[Unset, str] = UNSET
        if not isinstance(self.created_at, Unset):
            created_at = self.created_at.isoformat()

        cpu_usage_history: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.cpu_usage_history, Unset):
            cpu_usage_history = []
            for cpu_usage_history_item_data in self.cpu_usage_history:
                cpu_usage_history_item = cpu_usage_history_item_data.to_dict()
                cpu_usage_history.append(cpu_usage_history_item)

        memory_usage_history: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.memory_usage_history, Unset):
            memory_usage_history = []
            for memory_usage_history_item_data in self.memory_usage_history:
                memory_usage_history_item = memory_usage_history_item_data.to_dict()
                memory_usage_history.append(memory_usage_history_item)

        vram_usage_history: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.vram_usage_history, Unset):
            vram_usage_history = []
            for vram_usage_history_item_data in self.vram_usage_history:
                vram_usage_history_item = vram_usage_history_item_data.to_dict()
                vram_usage_history.append(vram_usage_history_item)

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
                "user_id": user_id,
            }
        )
        if application is not UNSET:
            field_dict["application"] = application
        if application_id is not UNSET:
            field_dict["application_id"] = application_id
        if node_name is not UNSET:
            field_dict["node_name"] = node_name
        if node_data_center is not UNSET:
            field_dict["node_data_center"] = node_data_center
        if started_at is not UNSET:
            field_dict["started_at"] = started_at
        if finished_at is not UNSET:
            field_dict["finished_at"] = finished_at
        if owner is not UNSET:
            field_dict["owner"] = owner
        if user_nickname is not UNSET:
            field_dict["user_nickname"] = user_nickname
        if job_source is not UNSET:
            field_dict["job_source"] = job_source
        if status is not UNSET:
            field_dict["status"] = status
        if created_at is not UNSET:
            field_dict["created_at"] = created_at
        if cpu_usage_history is not UNSET:
            field_dict["cpu_usage_history"] = cpu_usage_history
        if memory_usage_history is not UNSET:
            field_dict["memory_usage_history"] = memory_usage_history
        if vram_usage_history is not UNSET:
            field_dict["vram_usage_history"] = vram_usage_history
        if timing is not UNSET:
            field_dict["timing"] = timing
        if terminate_reason is not UNSET:
            field_dict["terminate_reason"] = terminate_reason

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.cpu_stats_by_time import CpuStatsByTime
        from ..models.memory_stats_by_time import MemoryStatsByTime
        from ..models.runner_timing_info import RunnerTimingInfo
        from ..models.vram_stats_by_time import VramStatsByTime

        d = src_dict.copy()
        runner_id = d.pop("runner_id")

        state = RunnerSearchResponseState(d.pop("state"))

        machine_type = d.pop("machine_type")

        region = d.pop("region")

        node_id = d.pop("node_id")

        user_id = d.pop("user_id")

        application = d.pop("application", UNSET)

        application_id = d.pop("application_id", UNSET)

        node_name = d.pop("node_name", UNSET)

        node_data_center = d.pop("node_data_center", UNSET)

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

        _owner = d.pop("owner", UNSET)
        owner: Union[Unset, RunnerSearchResponseOwner]
        if isinstance(_owner, Unset):
            owner = UNSET
        else:
            owner = RunnerSearchResponseOwner(_owner)

        user_nickname = d.pop("user_nickname", UNSET)

        job_source = d.pop("job_source", UNSET)

        status = d.pop("status", UNSET)

        _created_at = d.pop("created_at", UNSET)
        created_at: Union[Unset, datetime.datetime]
        if isinstance(_created_at, Unset):
            created_at = UNSET
        else:
            created_at = isoparse(_created_at)

        cpu_usage_history = []
        _cpu_usage_history = d.pop("cpu_usage_history", UNSET)
        for cpu_usage_history_item_data in _cpu_usage_history or []:
            cpu_usage_history_item = CpuStatsByTime.from_dict(cpu_usage_history_item_data)

            cpu_usage_history.append(cpu_usage_history_item)

        memory_usage_history = []
        _memory_usage_history = d.pop("memory_usage_history", UNSET)
        for memory_usage_history_item_data in _memory_usage_history or []:
            memory_usage_history_item = MemoryStatsByTime.from_dict(memory_usage_history_item_data)

            memory_usage_history.append(memory_usage_history_item)

        vram_usage_history = []
        _vram_usage_history = d.pop("vram_usage_history", UNSET)
        for vram_usage_history_item_data in _vram_usage_history or []:
            vram_usage_history_item = VramStatsByTime.from_dict(vram_usage_history_item_data)

            vram_usage_history.append(vram_usage_history_item)

        _timing = d.pop("timing", UNSET)
        timing: Union[Unset, RunnerTimingInfo]
        if isinstance(_timing, Unset):
            timing = UNSET
        else:
            timing = RunnerTimingInfo.from_dict(_timing)

        terminate_reason = d.pop("terminate_reason", UNSET)

        runner_search_response = cls(
            runner_id=runner_id,
            state=state,
            machine_type=machine_type,
            region=region,
            node_id=node_id,
            user_id=user_id,
            application=application,
            application_id=application_id,
            node_name=node_name,
            node_data_center=node_data_center,
            started_at=started_at,
            finished_at=finished_at,
            owner=owner,
            user_nickname=user_nickname,
            job_source=job_source,
            status=status,
            created_at=created_at,
            cpu_usage_history=cpu_usage_history,
            memory_usage_history=memory_usage_history,
            vram_usage_history=vram_usage_history,
            timing=timing,
            terminate_reason=terminate_reason,
        )

        runner_search_response.additional_properties = d
        return runner_search_response

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
