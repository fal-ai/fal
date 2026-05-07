from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.queue_info_by_user import QueueInfoByUser


T = TypeVar("T", bound="QueueInfo")


@_attrs_define
class QueueInfo:
    """
    Attributes:
        queue_size (int):
        in_progress (int):
        by_user (QueueInfoByUser):
    """

    queue_size: int
    in_progress: int
    by_user: "QueueInfoByUser"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        queue_size = self.queue_size

        in_progress = self.in_progress

        by_user = self.by_user.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "queue_size": queue_size,
                "in_progress": in_progress,
                "by_user": by_user,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.queue_info_by_user import QueueInfoByUser

        d = src_dict.copy()
        queue_size = d.pop("queue_size")

        in_progress = d.pop("in_progress")

        by_user = QueueInfoByUser.from_dict(d.pop("by_user"))

        queue_info = cls(
            queue_size=queue_size,
            in_progress=in_progress,
            by_user=by_user,
        )

        queue_info.additional_properties = d
        return queue_info

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
