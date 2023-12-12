import datetime
from typing import Any, Dict, List, Type, TypeVar

import attr
from dateutil.parser import isoparse

T = TypeVar("T", bound="UsagePerMachineType")


@attr.s(auto_attribs=True)
class UsagePerMachineType:
    """
    Attributes:
        machine_type (str):
        day (datetime.date):
        num_workers (int):
        num_runs (int):
        run_call_seconds (int):
        run_execution_seconds (int):
        run_queue_seconds (int):
        worker_uptime_seconds (int):
        worker_idle_seconds (int):
    """

    machine_type: str
    day: datetime.date
    num_workers: int
    num_runs: int
    run_call_seconds: int
    run_execution_seconds: int
    run_queue_seconds: int
    worker_uptime_seconds: int
    worker_idle_seconds: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        machine_type = self.machine_type
        day = self.day.isoformat()
        num_workers = self.num_workers
        num_runs = self.num_runs
        run_call_seconds = self.run_call_seconds
        run_execution_seconds = self.run_execution_seconds
        run_queue_seconds = self.run_queue_seconds
        worker_uptime_seconds = self.worker_uptime_seconds
        worker_idle_seconds = self.worker_idle_seconds

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "machine_type": machine_type,
                "day": day,
                "num_workers": num_workers,
                "num_runs": num_runs,
                "run_call_seconds": run_call_seconds,
                "run_execution_seconds": run_execution_seconds,
                "run_queue_seconds": run_queue_seconds,
                "worker_uptime_seconds": worker_uptime_seconds,
                "worker_idle_seconds": worker_idle_seconds,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        machine_type = d.pop("machine_type")

        day = isoparse(d.pop("day")).date()

        num_workers = d.pop("num_workers")

        num_runs = d.pop("num_runs")

        run_call_seconds = d.pop("run_call_seconds")

        run_execution_seconds = d.pop("run_execution_seconds")

        run_queue_seconds = d.pop("run_queue_seconds")

        worker_uptime_seconds = d.pop("worker_uptime_seconds")

        worker_idle_seconds = d.pop("worker_idle_seconds")

        usage_per_machine_type = cls(
            machine_type=machine_type,
            day=day,
            num_workers=num_workers,
            num_runs=num_runs,
            run_call_seconds=run_call_seconds,
            run_execution_seconds=run_execution_seconds,
            run_queue_seconds=run_queue_seconds,
            worker_uptime_seconds=worker_uptime_seconds,
            worker_idle_seconds=worker_idle_seconds,
        )

        usage_per_machine_type.additional_properties = d
        return usage_per_machine_type

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
