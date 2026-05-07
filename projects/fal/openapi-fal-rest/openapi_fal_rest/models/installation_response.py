from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.vercel_installation_plan import VercelInstallationPlan


T = TypeVar("T", bound="InstallationResponse")


@_attrs_define
class InstallationResponse:
    """
    Attributes:
        billing_plan (VercelInstallationPlan):
    """

    billing_plan: "VercelInstallationPlan"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        billing_plan = self.billing_plan.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "billingPlan": billing_plan,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.vercel_installation_plan import VercelInstallationPlan

        d = src_dict.copy()
        billing_plan = VercelInstallationPlan.from_dict(d.pop("billingPlan"))

        installation_response = cls(
            billing_plan=billing_plan,
        )

        installation_response.additional_properties = d
        return installation_response

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
