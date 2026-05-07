from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.validation_result_code import ValidationResultCode
from ..models.validation_result_severity import ValidationResultSeverity
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.validation_recommendation import ValidationRecommendation


T = TypeVar("T", bound="ValidationResult")


@_attrs_define
class ValidationResult:
    """A single validation result with severity.

    Attributes:
        code (ValidationResultCode):
        severity (ValidationResultSeverity):
        message (str):
        details (Union[Unset, str]):
        recommendation (Union[Unset, ValidationRecommendation]): A recommended corrective action for a validation issue.
    """

    code: ValidationResultCode
    severity: ValidationResultSeverity
    message: str
    details: Union[Unset, str] = UNSET
    recommendation: Union[Unset, "ValidationRecommendation"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        code = self.code.value

        severity = self.severity.value

        message = self.message

        details = self.details

        recommendation: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.recommendation, Unset):
            recommendation = self.recommendation.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "code": code,
                "severity": severity,
                "message": message,
            }
        )
        if details is not UNSET:
            field_dict["details"] = details
        if recommendation is not UNSET:
            field_dict["recommendation"] = recommendation

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.validation_recommendation import ValidationRecommendation

        d = src_dict.copy()
        code = ValidationResultCode(d.pop("code"))

        severity = ValidationResultSeverity(d.pop("severity"))

        message = d.pop("message")

        details = d.pop("details", UNSET)

        _recommendation = d.pop("recommendation", UNSET)
        recommendation: Union[Unset, ValidationRecommendation]
        if isinstance(_recommendation, Unset):
            recommendation = UNSET
        else:
            recommendation = ValidationRecommendation.from_dict(_recommendation)

        validation_result = cls(
            code=code,
            severity=severity,
            message=message,
            details=details,
            recommendation=recommendation,
        )

        validation_result.additional_properties = d
        return validation_result

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
