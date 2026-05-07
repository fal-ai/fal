from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.vercel_installation_plan import VercelInstallationPlan


T = TypeVar("T", bound="DefaultPlanResponse")


@_attrs_define
class DefaultPlanResponse:
    """
    Attributes:
        plans (list['VercelInstallationPlan']):
    """

    plans: list["VercelInstallationPlan"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        plans = []
        for plans_item_data in self.plans:
            plans_item = plans_item_data.to_dict()
            plans.append(plans_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "plans": plans,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.vercel_installation_plan import VercelInstallationPlan

        d = src_dict.copy()
        plans = []
        _plans = d.pop("plans")
        for plans_item_data in _plans:
            plans_item = VercelInstallationPlan.from_dict(plans_item_data)

            plans.append(plans_item)

        default_plan_response = cls(
            plans=plans,
        )

        default_plan_response.additional_properties = d
        return default_plan_response

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
