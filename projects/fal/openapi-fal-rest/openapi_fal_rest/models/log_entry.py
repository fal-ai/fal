import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.log_entry_labels import LogEntryLabels


T = TypeVar("T", bound="LogEntry")


@_attrs_define
class LogEntry:
    """
    Attributes:
        timestamp (datetime.datetime):
        level (str):
        message (str):
        app (str):
        revision (str):
        labels (Union[Unset, LogEntryLabels]):
    """

    timestamp: datetime.datetime
    level: str
    message: str
    app: str
    revision: str
    labels: Union[Unset, "LogEntryLabels"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        timestamp = self.timestamp.isoformat()

        level = self.level

        message = self.message

        app = self.app

        revision = self.revision

        labels: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.labels, Unset):
            labels = self.labels.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "timestamp": timestamp,
                "level": level,
                "message": message,
                "app": app,
                "revision": revision,
            }
        )
        if labels is not UNSET:
            field_dict["labels"] = labels

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.log_entry_labels import LogEntryLabels

        d = src_dict.copy()
        timestamp = isoparse(d.pop("timestamp"))

        level = d.pop("level")

        message = d.pop("message")

        app = d.pop("app")

        revision = d.pop("revision")

        _labels = d.pop("labels", UNSET)
        labels: Union[Unset, LogEntryLabels]
        if isinstance(_labels, Unset):
            labels = UNSET
        else:
            labels = LogEntryLabels.from_dict(_labels)

        log_entry = cls(
            timestamp=timestamp,
            level=level,
            message=message,
            app=app,
            revision=revision,
            labels=labels,
        )

        log_entry.additional_properties = d
        return log_entry

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
