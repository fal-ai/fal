from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.user_concurrency_by_time import UserConcurrencyByTime


T = TypeVar("T", bound="UserConcurrencyHistory")


@_attrs_define
class UserConcurrencyHistory:
    """
    Attributes:
        user_id (str):
        time_stats (list['UserConcurrencyByTime']):
    """

    user_id: str
    time_stats: list["UserConcurrencyByTime"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        time_stats = []
        for time_stats_item_data in self.time_stats:
            time_stats_item = time_stats_item_data.to_dict()
            time_stats.append(time_stats_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "time_stats": time_stats,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.user_concurrency_by_time import UserConcurrencyByTime

        d = src_dict.copy()
        user_id = d.pop("user_id")

        time_stats = []
        _time_stats = d.pop("time_stats")
        for time_stats_item_data in _time_stats:
            time_stats_item = UserConcurrencyByTime.from_dict(time_stats_item_data)

            time_stats.append(time_stats_item)

        user_concurrency_history = cls(
            user_id=user_id,
            time_stats=time_stats,
        )

        user_concurrency_history.additional_properties = d
        return user_concurrency_history

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
