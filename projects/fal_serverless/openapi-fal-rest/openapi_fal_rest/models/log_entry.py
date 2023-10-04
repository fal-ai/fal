import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.log_entry_labels import LogEntryLabels


T = TypeVar("T", bound="LogEntry")


@attr.s(auto_attribs=True)
class LogEntry:
    """
    Attributes:
        timestamp (datetime.datetime):
        level (str):
        message (str):
        app (str):
        labels (Union[Unset, LogEntryLabels]):
    """

    timestamp: datetime.datetime
    level: str
    message: str
    app: str
    labels: Union[Unset, "LogEntryLabels"] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        timestamp = self.timestamp.isoformat()

        level = self.level
        message = self.message
        app = self.app
        labels: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.labels, Unset):
            labels = self.labels.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "timestamp": timestamp,
                "level": level,
                "message": message,
                "app": app,
            }
        )
        if labels is not UNSET:
            field_dict["labels"] = labels

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.log_entry_labels import LogEntryLabels

        d = src_dict.copy()
        timestamp = isoparse(d.pop("timestamp"))

        level = d.pop("level")

        message = d.pop("message")

        app = d.pop("app")

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
            labels=labels,
        )

        log_entry.additional_properties = d
        return log_entry

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
