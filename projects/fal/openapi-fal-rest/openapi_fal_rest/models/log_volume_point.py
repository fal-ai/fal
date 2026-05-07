import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="LogVolumePoint")


@_attrs_define
class LogVolumePoint:
    """A single data point representing log counts at a specific timestamp.

    Attributes:
        timestamp (datetime.datetime):
        trace (Union[Unset, int]):  Default: 0.
        debug (Union[Unset, int]):  Default: 0.
        info (Union[Unset, int]):  Default: 0.
        warning (Union[Unset, int]):  Default: 0.
        error (Union[Unset, int]):  Default: 0.
        stdout (Union[Unset, int]):  Default: 0.
        stderr (Union[Unset, int]):  Default: 0.
    """

    timestamp: datetime.datetime
    trace: Union[Unset, int] = 0
    debug: Union[Unset, int] = 0
    info: Union[Unset, int] = 0
    warning: Union[Unset, int] = 0
    error: Union[Unset, int] = 0
    stdout: Union[Unset, int] = 0
    stderr: Union[Unset, int] = 0
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        timestamp = self.timestamp.isoformat()

        trace = self.trace

        debug = self.debug

        info = self.info

        warning = self.warning

        error = self.error

        stdout = self.stdout

        stderr = self.stderr

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "timestamp": timestamp,
            }
        )
        if trace is not UNSET:
            field_dict["trace"] = trace
        if debug is not UNSET:
            field_dict["debug"] = debug
        if info is not UNSET:
            field_dict["info"] = info
        if warning is not UNSET:
            field_dict["warning"] = warning
        if error is not UNSET:
            field_dict["error"] = error
        if stdout is not UNSET:
            field_dict["stdout"] = stdout
        if stderr is not UNSET:
            field_dict["stderr"] = stderr

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        timestamp = isoparse(d.pop("timestamp"))

        trace = d.pop("trace", UNSET)

        debug = d.pop("debug", UNSET)

        info = d.pop("info", UNSET)

        warning = d.pop("warning", UNSET)

        error = d.pop("error", UNSET)

        stdout = d.pop("stdout", UNSET)

        stderr = d.pop("stderr", UNSET)

        log_volume_point = cls(
            timestamp=timestamp,
            trace=trace,
            debug=debug,
            info=info,
            warning=warning,
            error=error,
            stdout=stdout,
            stderr=stderr,
        )

        log_volume_point.additional_properties = d
        return log_volume_point

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
