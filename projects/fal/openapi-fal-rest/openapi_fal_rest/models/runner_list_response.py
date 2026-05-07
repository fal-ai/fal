from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.runner_info import RunnerInfo


T = TypeVar("T", bound="RunnerListResponse")


@_attrs_define
class RunnerListResponse:
    """
    Attributes:
        runners (list['RunnerInfo']):
    """

    runners: list["RunnerInfo"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        runners = []
        for runners_item_data in self.runners:
            runners_item = runners_item_data.to_dict()
            runners.append(runners_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "runners": runners,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.runner_info import RunnerInfo

        d = src_dict.copy()
        runners = []
        _runners = d.pop("runners")
        for runners_item_data in _runners:
            runners_item = RunnerInfo.from_dict(runners_item_data)

            runners.append(runners_item)

        runner_list_response = cls(
            runners=runners,
        )

        runner_list_response.additional_properties = d
        return runner_list_response

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
