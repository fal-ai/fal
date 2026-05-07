from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.enriched_team_info import EnrichedTeamInfo
    from ..models.team_usage_endpoints import TeamUsageEndpoints


T = TypeVar("T", bound="TeamUsage")


@_attrs_define
class TeamUsage:
    """Usage summary for a single team account (for org team breakdown).

    Attributes:
        user_id (str):
        total_amount (float):
        endpoints (TeamUsageEndpoints):
        team_info (Union[Unset, EnrichedTeamInfo]): Database information about a team account for invoice enrichment.
    """

    user_id: str
    total_amount: float
    endpoints: "TeamUsageEndpoints"
    team_info: Union[Unset, "EnrichedTeamInfo"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        total_amount = self.total_amount

        endpoints = self.endpoints.to_dict()

        team_info: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.team_info, Unset):
            team_info = self.team_info.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "total_amount": total_amount,
                "endpoints": endpoints,
            }
        )
        if team_info is not UNSET:
            field_dict["team_info"] = team_info

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.enriched_team_info import EnrichedTeamInfo
        from ..models.team_usage_endpoints import TeamUsageEndpoints

        d = src_dict.copy()
        user_id = d.pop("user_id")

        total_amount = d.pop("total_amount")

        endpoints = TeamUsageEndpoints.from_dict(d.pop("endpoints"))

        _team_info = d.pop("team_info", UNSET)
        team_info: Union[Unset, EnrichedTeamInfo]
        if isinstance(_team_info, Unset):
            team_info = UNSET
        else:
            team_info = EnrichedTeamInfo.from_dict(_team_info)

        team_usage = cls(
            user_id=user_id,
            total_amount=total_amount,
            endpoints=endpoints,
            team_info=team_info,
        )

        team_usage.additional_properties = d
        return team_usage

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
