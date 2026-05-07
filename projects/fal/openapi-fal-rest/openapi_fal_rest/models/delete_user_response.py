from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="DeleteUserResponse")


@_attrs_define
class DeleteUserResponse:
    """
    Attributes:
        user_id (str):
        email (str):
        errors (list[str]):
        explanation (str):
    """

    user_id: str
    email: str
    errors: list[str]
    explanation: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        email = self.email

        errors = self.errors

        explanation = self.explanation

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "email": email,
                "errors": errors,
                "explanation": explanation,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        email = d.pop("email")

        errors = cast(list[str], d.pop("errors"))

        explanation = d.pop("explanation")

        delete_user_response = cls(
            user_id=user_id,
            email=email,
            errors=errors,
            explanation=explanation,
        )

        delete_user_response.additional_properties = d
        return delete_user_response

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
