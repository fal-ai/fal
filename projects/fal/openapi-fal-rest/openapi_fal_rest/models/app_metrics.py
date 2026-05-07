from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.app_metrics_cpu_count_by_type import AppMetricsCpuCountByType
    from ..models.app_metrics_gpu_count_by_type import AppMetricsGpuCountByType


T = TypeVar("T", bound="AppMetrics")


@_attrs_define
class AppMetrics:
    """
    Attributes:
        runner_count (int):
        gpu_count (int):
        queue_size (int):
        gpu_count_by_type (Union[Unset, AppMetricsGpuCountByType]):
        cpu_count (Union[Unset, int]):  Default: 0.
        cpu_count_by_type (Union[Unset, AppMetricsCpuCountByType]):
    """

    runner_count: int
    gpu_count: int
    queue_size: int
    gpu_count_by_type: Union[Unset, "AppMetricsGpuCountByType"] = UNSET
    cpu_count: Union[Unset, int] = 0
    cpu_count_by_type: Union[Unset, "AppMetricsCpuCountByType"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        runner_count = self.runner_count

        gpu_count = self.gpu_count

        queue_size = self.queue_size

        gpu_count_by_type: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.gpu_count_by_type, Unset):
            gpu_count_by_type = self.gpu_count_by_type.to_dict()

        cpu_count = self.cpu_count

        cpu_count_by_type: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.cpu_count_by_type, Unset):
            cpu_count_by_type = self.cpu_count_by_type.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "runner_count": runner_count,
                "gpu_count": gpu_count,
                "queue_size": queue_size,
            }
        )
        if gpu_count_by_type is not UNSET:
            field_dict["gpu_count_by_type"] = gpu_count_by_type
        if cpu_count is not UNSET:
            field_dict["cpu_count"] = cpu_count
        if cpu_count_by_type is not UNSET:
            field_dict["cpu_count_by_type"] = cpu_count_by_type

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.app_metrics_cpu_count_by_type import AppMetricsCpuCountByType
        from ..models.app_metrics_gpu_count_by_type import AppMetricsGpuCountByType

        d = src_dict.copy()
        runner_count = d.pop("runner_count")

        gpu_count = d.pop("gpu_count")

        queue_size = d.pop("queue_size")

        _gpu_count_by_type = d.pop("gpu_count_by_type", UNSET)
        gpu_count_by_type: Union[Unset, AppMetricsGpuCountByType]
        if isinstance(_gpu_count_by_type, Unset):
            gpu_count_by_type = UNSET
        else:
            gpu_count_by_type = AppMetricsGpuCountByType.from_dict(_gpu_count_by_type)

        cpu_count = d.pop("cpu_count", UNSET)

        _cpu_count_by_type = d.pop("cpu_count_by_type", UNSET)
        cpu_count_by_type: Union[Unset, AppMetricsCpuCountByType]
        if isinstance(_cpu_count_by_type, Unset):
            cpu_count_by_type = UNSET
        else:
            cpu_count_by_type = AppMetricsCpuCountByType.from_dict(_cpu_count_by_type)

        app_metrics = cls(
            runner_count=runner_count,
            gpu_count=gpu_count,
            queue_size=queue_size,
            gpu_count_by_type=gpu_count_by_type,
            cpu_count=cpu_count,
            cpu_count_by_type=cpu_count_by_type,
        )

        app_metrics.additional_properties = d
        return app_metrics

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
