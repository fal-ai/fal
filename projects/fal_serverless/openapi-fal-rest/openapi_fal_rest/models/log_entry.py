from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="LogEntry")


@attr.s(auto_attribs=True)
class LogEntry:
    """
    Attributes:
        timestamp (str):
        level (str):
        message (str):
        app (str):
    """

    timestamp: str
    level: str
    message: str
    app: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        timestamp = self.timestamp
        level = self.level
        message = self.message
        app = self.app

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

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        timestamp = d.pop("timestamp")

        level = d.pop("level")

        message = d.pop("message")

        app = d.pop("app")

        log_entry = cls(
            timestamp=timestamp,
            level=level,
            message=message,
            app=app,
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
