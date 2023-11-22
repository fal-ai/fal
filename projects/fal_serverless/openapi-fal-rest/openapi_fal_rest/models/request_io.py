import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

import attr
from dateutil.parser import isoparse

if TYPE_CHECKING:
    from ..models.request_io_json_input import RequestIOJsonInput
    from ..models.request_io_json_output import RequestIOJsonOutput


T = TypeVar("T", bound="RequestIO")


@attr.s(auto_attribs=True)
class RequestIO:
    """
    Attributes:
        request_id (str):
        date (datetime.datetime):
        json_input (RequestIOJsonInput):
        json_output (RequestIOJsonOutput):
        status_code (int):
        logs (str):
        duration_in_seconds (int):
    """

    request_id: str
    date: datetime.datetime
    json_input: "RequestIOJsonInput"
    json_output: "RequestIOJsonOutput"
    status_code: int
    logs: str
    duration_in_seconds: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        request_id = self.request_id
        date = self.date.isoformat()

        json_input = self.json_input.to_dict()

        json_output = self.json_output.to_dict()

        status_code = self.status_code
        logs = self.logs
        duration_in_seconds = self.duration_in_seconds

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "request_id": request_id,
                "date": date,
                "json_input": json_input,
                "json_output": json_output,
                "status_code": status_code,
                "logs": logs,
                "duration_in_seconds": duration_in_seconds,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.request_io_json_input import RequestIOJsonInput
        from ..models.request_io_json_output import RequestIOJsonOutput

        d = src_dict.copy()
        request_id = d.pop("request_id")

        date = isoparse(d.pop("date"))

        json_input = RequestIOJsonInput.from_dict(d.pop("json_input"))

        json_output = RequestIOJsonOutput.from_dict(d.pop("json_output"))

        status_code = d.pop("status_code")

        logs = d.pop("logs")

        duration_in_seconds = d.pop("duration_in_seconds")

        request_io = cls(
            request_id=request_id,
            date=date,
            json_input=json_input,
            json_output=json_output,
            status_code=status_code,
            logs=logs,
            duration_in_seconds=duration_in_seconds,
        )

        request_io.additional_properties = d
        return request_io

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
