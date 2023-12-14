from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="GroupedUsageDetail")


@attr.s(auto_attribs=True)
class GroupedUsageDetail:
    """
    Attributes:
        model_id (str):
        machine_type (str):
        request_count (int):
        median_duration (float):
        total_duration (float):
    """

    model_id: str
    machine_type: str
    request_count: int
    median_duration: float
    total_duration: float
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        model_id = self.model_id
        machine_type = self.machine_type
        request_count = self.request_count
        median_duration = self.median_duration
        total_duration = self.total_duration

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "model_id": model_id,
                "machine_type": machine_type,
                "request_count": request_count,
                "median_duration": median_duration,
                "total_duration": total_duration,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        model_id = d.pop("model_id")

        machine_type = d.pop("machine_type")

        request_count = d.pop("request_count")

        median_duration = d.pop("median_duration")

        total_duration = d.pop("total_duration")

        grouped_usage_detail = cls(
            model_id=model_id,
            machine_type=machine_type,
            request_count=request_count,
            median_duration=median_duration,
            total_duration=total_duration,
        )

        grouped_usage_detail.additional_properties = d
        return grouped_usage_detail

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
