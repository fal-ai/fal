from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.model_degradation_history import ModelDegradationHistory


T = TypeVar("T", bound="ModelStatusInfo")


@_attrs_define
class ModelStatusInfo:
    """
    Attributes:
        application (str):
        display_name (str):
        degradations (list['ModelDegradationHistory']):
    """

    application: str
    display_name: str
    degradations: list["ModelDegradationHistory"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        application = self.application

        display_name = self.display_name

        degradations = []
        for degradations_item_data in self.degradations:
            degradations_item = degradations_item_data.to_dict()
            degradations.append(degradations_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "application": application,
                "display_name": display_name,
                "degradations": degradations,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.model_degradation_history import ModelDegradationHistory

        d = src_dict.copy()
        application = d.pop("application")

        display_name = d.pop("display_name")

        degradations = []
        _degradations = d.pop("degradations")
        for degradations_item_data in _degradations:
            degradations_item = ModelDegradationHistory.from_dict(degradations_item_data)

            degradations.append(degradations_item)

        model_status_info = cls(
            application=application,
            display_name=display_name,
            degradations=degradations,
        )

        model_status_info.additional_properties = d
        return model_status_info

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
