from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.validation_recommendation_action import ValidationRecommendationAction

T = TypeVar("T", bound="ValidationRecommendation")


@_attrs_define
class ValidationRecommendation:
    """A recommended corrective action for a validation issue.

    Attributes:
        action (ValidationRecommendationAction):
        description (str):
    """

    action: ValidationRecommendationAction
    description: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        action = self.action.value

        description = self.description

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "action": action,
                "description": description,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        action = ValidationRecommendationAction(d.pop("action"))

        description = d.pop("description")

        validation_recommendation = cls(
            action=action,
            description=description,
        )

        validation_recommendation.additional_properties = d
        return validation_recommendation

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
