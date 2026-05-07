from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.user_metrics_summary_total_cpu_by_type import UserMetricsSummaryTotalCpuByType
    from ..models.user_metrics_summary_total_gpu_by_type import UserMetricsSummaryTotalGpuByType


T = TypeVar("T", bound="UserMetricsSummary")


@_attrs_define
class UserMetricsSummary:
    """
    Attributes:
        total_runners (int):
        total_gpus (int):
        total_queue_size (int):
        total_gpu_by_type (Union[Unset, UserMetricsSummaryTotalGpuByType]):
        total_cpus (Union[Unset, int]):  Default: 0.
        total_cpu_by_type (Union[Unset, UserMetricsSummaryTotalCpuByType]):
    """

    total_runners: int
    total_gpus: int
    total_queue_size: int
    total_gpu_by_type: Union[Unset, "UserMetricsSummaryTotalGpuByType"] = UNSET
    total_cpus: Union[Unset, int] = 0
    total_cpu_by_type: Union[Unset, "UserMetricsSummaryTotalCpuByType"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        total_runners = self.total_runners

        total_gpus = self.total_gpus

        total_queue_size = self.total_queue_size

        total_gpu_by_type: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.total_gpu_by_type, Unset):
            total_gpu_by_type = self.total_gpu_by_type.to_dict()

        total_cpus = self.total_cpus

        total_cpu_by_type: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.total_cpu_by_type, Unset):
            total_cpu_by_type = self.total_cpu_by_type.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "total_runners": total_runners,
                "total_gpus": total_gpus,
                "total_queue_size": total_queue_size,
            }
        )
        if total_gpu_by_type is not UNSET:
            field_dict["total_gpu_by_type"] = total_gpu_by_type
        if total_cpus is not UNSET:
            field_dict["total_cpus"] = total_cpus
        if total_cpu_by_type is not UNSET:
            field_dict["total_cpu_by_type"] = total_cpu_by_type

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.user_metrics_summary_total_cpu_by_type import UserMetricsSummaryTotalCpuByType
        from ..models.user_metrics_summary_total_gpu_by_type import UserMetricsSummaryTotalGpuByType

        d = src_dict.copy()
        total_runners = d.pop("total_runners")

        total_gpus = d.pop("total_gpus")

        total_queue_size = d.pop("total_queue_size")

        _total_gpu_by_type = d.pop("total_gpu_by_type", UNSET)
        total_gpu_by_type: Union[Unset, UserMetricsSummaryTotalGpuByType]
        if isinstance(_total_gpu_by_type, Unset):
            total_gpu_by_type = UNSET
        else:
            total_gpu_by_type = UserMetricsSummaryTotalGpuByType.from_dict(_total_gpu_by_type)

        total_cpus = d.pop("total_cpus", UNSET)

        _total_cpu_by_type = d.pop("total_cpu_by_type", UNSET)
        total_cpu_by_type: Union[Unset, UserMetricsSummaryTotalCpuByType]
        if isinstance(_total_cpu_by_type, Unset):
            total_cpu_by_type = UNSET
        else:
            total_cpu_by_type = UserMetricsSummaryTotalCpuByType.from_dict(_total_cpu_by_type)

        user_metrics_summary = cls(
            total_runners=total_runners,
            total_gpus=total_gpus,
            total_queue_size=total_queue_size,
            total_gpu_by_type=total_gpu_by_type,
            total_cpus=total_cpus,
            total_cpu_by_type=total_cpu_by_type,
        )

        user_metrics_summary.additional_properties = d
        return user_metrics_summary

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
