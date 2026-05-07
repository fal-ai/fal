from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="LogDrainTestResult")


@_attrs_define
class LogDrainTestResult:
    """
    Attributes:
        success (bool):
        status_code (Union[Unset, int]):
        error_message (Union[Unset, str]):
    """

    success: bool
    status_code: Union[Unset, int] = UNSET
    error_message: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        success = self.success

        status_code = self.status_code

        error_message = self.error_message

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "success": success,
            }
        )
        if status_code is not UNSET:
            field_dict["status_code"] = status_code
        if error_message is not UNSET:
            field_dict["error_message"] = error_message

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        success = d.pop("success")

        status_code = d.pop("status_code", UNSET)

        error_message = d.pop("error_message", UNSET)

        log_drain_test_result = cls(
            success=success,
            status_code=status_code,
            error_message=error_message,
        )

        log_drain_test_result.additional_properties = d
        return log_drain_test_result

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
