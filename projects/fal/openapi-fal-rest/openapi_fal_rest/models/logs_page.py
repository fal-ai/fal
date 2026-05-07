from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.log_entry import LogEntry
    from ..models.logs_page_stats_item import LogsPageStatsItem


T = TypeVar("T", bound="LogsPage")


@_attrs_define
class LogsPage:
    """
    Attributes:
        items (list['LogEntry']):
        next_until (str):
        next_since (Union[Unset, str]):
        stats (Union[Unset, list['LogsPageStatsItem']]):
    """

    items: list["LogEntry"]
    next_until: str
    next_since: Union[Unset, str] = UNSET
    stats: Union[Unset, list["LogsPageStatsItem"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        items = []
        for items_item_data in self.items:
            items_item = items_item_data.to_dict()
            items.append(items_item)

        next_until = self.next_until

        next_since = self.next_since

        stats: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.stats, Unset):
            stats = []
            for stats_item_data in self.stats:
                stats_item = stats_item_data.to_dict()
                stats.append(stats_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "items": items,
                "next_until": next_until,
            }
        )
        if next_since is not UNSET:
            field_dict["next_since"] = next_since
        if stats is not UNSET:
            field_dict["stats"] = stats

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.log_entry import LogEntry
        from ..models.logs_page_stats_item import LogsPageStatsItem

        d = src_dict.copy()
        items = []
        _items = d.pop("items")
        for items_item_data in _items:
            items_item = LogEntry.from_dict(items_item_data)

            items.append(items_item)

        next_until = d.pop("next_until")

        next_since = d.pop("next_since", UNSET)

        stats = []
        _stats = d.pop("stats", UNSET)
        for stats_item_data in _stats or []:
            stats_item = LogsPageStatsItem.from_dict(stats_item_data)

            stats.append(stats_item)

        logs_page = cls(
            items=items,
            next_until=next_until,
            next_since=next_since,
            stats=stats,
        )

        logs_page.additional_properties = d
        return logs_page

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
