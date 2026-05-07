import datetime
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.model_status import ModelStatus

T = TypeVar("T", bound="ModelDegradationHistory")


@_attrs_define
class ModelDegradationHistory:
    """
    Attributes:
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        status (ModelStatus):
        average_error_rate (float):
    """

    start_time: datetime.datetime
    end_time: datetime.datetime
    status: ModelStatus
    average_error_rate: float
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        start_time = self.start_time.isoformat()

        end_time = self.end_time.isoformat()

        status = self.status.value

        average_error_rate = self.average_error_rate

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "start_time": start_time,
                "end_time": end_time,
                "status": status,
                "average_error_rate": average_error_rate,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        start_time = isoparse(d.pop("start_time"))

        end_time = isoparse(d.pop("end_time"))

        status = ModelStatus(d.pop("status"))

        average_error_rate = d.pop("average_error_rate")

        model_degradation_history = cls(
            start_time=start_time,
            end_time=end_time,
            status=status,
            average_error_rate=average_error_rate,
        )

        model_degradation_history.additional_properties = d
        return model_degradation_history

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
