from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="EnrichedTeamInfo")


@_attrs_define
class EnrichedTeamInfo:
    """Database information about a team account for invoice enrichment.

    Attributes:
        user_id (str):
        nickname (str):
        full_name (str):
    """

    user_id: str
    nickname: str
    full_name: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        nickname = self.nickname

        full_name = self.full_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "nickname": nickname,
                "full_name": full_name,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        nickname = d.pop("nickname")

        full_name = d.pop("full_name")

        enriched_team_info = cls(
            user_id=user_id,
            nickname=nickname,
            full_name=full_name,
        )

        enriched_team_info.additional_properties = d
        return enriched_team_info

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
