from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ExtraPermissions")


@_attrs_define
class ExtraPermissions:
    """
    Attributes:
        compute (Union[Unset, bool]):  Default: False.
        beta_access (Union[Unset, bool]):  Default: False.
        pricing_access (Union[Unset, bool]):  Default: False.
        spending_alerts (Union[Unset, bool]):  Default: False.
        spending_alerts_v2 (Union[Unset, bool]):  Default: False.
        spending_locks (Union[Unset, bool]):  Default: False.
        focus_reports (Union[Unset, bool]):  Default: False.
        enriched_invoices (Union[Unset, bool]):  Default: False.
        endpoint_access_controls (Union[Unset, bool]):  Default: False.
        enable_per_user_auth (Union[Unset, bool]):  Default: False.
        model_status_access (Union[Unset, bool]):  Default: False.
        is_early_access (Union[Unset, bool]):  Default: False.
    """

    compute: Union[Unset, bool] = False
    beta_access: Union[Unset, bool] = False
    pricing_access: Union[Unset, bool] = False
    spending_alerts: Union[Unset, bool] = False
    spending_alerts_v2: Union[Unset, bool] = False
    spending_locks: Union[Unset, bool] = False
    focus_reports: Union[Unset, bool] = False
    enriched_invoices: Union[Unset, bool] = False
    endpoint_access_controls: Union[Unset, bool] = False
    enable_per_user_auth: Union[Unset, bool] = False
    model_status_access: Union[Unset, bool] = False
    is_early_access: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        compute = self.compute

        beta_access = self.beta_access

        pricing_access = self.pricing_access

        spending_alerts = self.spending_alerts

        spending_alerts_v2 = self.spending_alerts_v2

        spending_locks = self.spending_locks

        focus_reports = self.focus_reports

        enriched_invoices = self.enriched_invoices

        endpoint_access_controls = self.endpoint_access_controls

        enable_per_user_auth = self.enable_per_user_auth

        model_status_access = self.model_status_access

        is_early_access = self.is_early_access

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if compute is not UNSET:
            field_dict["compute"] = compute
        if beta_access is not UNSET:
            field_dict["beta_access"] = beta_access
        if pricing_access is not UNSET:
            field_dict["pricing_access"] = pricing_access
        if spending_alerts is not UNSET:
            field_dict["spending_alerts"] = spending_alerts
        if spending_alerts_v2 is not UNSET:
            field_dict["spending_alerts_v2"] = spending_alerts_v2
        if spending_locks is not UNSET:
            field_dict["spending_locks"] = spending_locks
        if focus_reports is not UNSET:
            field_dict["focus_reports"] = focus_reports
        if enriched_invoices is not UNSET:
            field_dict["enriched_invoices"] = enriched_invoices
        if endpoint_access_controls is not UNSET:
            field_dict["endpoint_access_controls"] = endpoint_access_controls
        if enable_per_user_auth is not UNSET:
            field_dict["enable_per_user_auth"] = enable_per_user_auth
        if model_status_access is not UNSET:
            field_dict["model_status_access"] = model_status_access
        if is_early_access is not UNSET:
            field_dict["is_early_access"] = is_early_access

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        compute = d.pop("compute", UNSET)

        beta_access = d.pop("beta_access", UNSET)

        pricing_access = d.pop("pricing_access", UNSET)

        spending_alerts = d.pop("spending_alerts", UNSET)

        spending_alerts_v2 = d.pop("spending_alerts_v2", UNSET)

        spending_locks = d.pop("spending_locks", UNSET)

        focus_reports = d.pop("focus_reports", UNSET)

        enriched_invoices = d.pop("enriched_invoices", UNSET)

        endpoint_access_controls = d.pop("endpoint_access_controls", UNSET)

        enable_per_user_auth = d.pop("enable_per_user_auth", UNSET)

        model_status_access = d.pop("model_status_access", UNSET)

        is_early_access = d.pop("is_early_access", UNSET)

        extra_permissions = cls(
            compute=compute,
            beta_access=beta_access,
            pricing_access=pricing_access,
            spending_alerts=spending_alerts,
            spending_alerts_v2=spending_alerts_v2,
            spending_locks=spending_locks,
            focus_reports=focus_reports,
            enriched_invoices=enriched_invoices,
            endpoint_access_controls=endpoint_access_controls,
            enable_per_user_auth=enable_per_user_auth,
            model_status_access=model_status_access,
            is_early_access=is_early_access,
        )

        extra_permissions.additional_properties = d
        return extra_permissions

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
