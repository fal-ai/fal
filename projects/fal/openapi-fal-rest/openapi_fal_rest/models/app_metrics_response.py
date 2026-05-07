from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.app_metrics_response_apps import AppMetricsResponseApps
    from ..models.user_metrics_summary import UserMetricsSummary


T = TypeVar("T", bound="AppMetricsResponse")


@_attrs_define
class AppMetricsResponse:
    """
    Attributes:
        summary (UserMetricsSummary):
        apps (Union[Unset, AppMetricsResponseApps]):
    """

    summary: "UserMetricsSummary"
    apps: Union[Unset, "AppMetricsResponseApps"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        summary = self.summary.to_dict()

        apps: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.apps, Unset):
            apps = self.apps.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "summary": summary,
            }
        )
        if apps is not UNSET:
            field_dict["apps"] = apps

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.app_metrics_response_apps import AppMetricsResponseApps
        from ..models.user_metrics_summary import UserMetricsSummary

        d = src_dict.copy()
        summary = UserMetricsSummary.from_dict(d.pop("summary"))

        _apps = d.pop("apps", UNSET)
        apps: Union[Unset, AppMetricsResponseApps]
        if isinstance(_apps, Unset):
            apps = UNSET
        else:
            apps = AppMetricsResponseApps.from_dict(_apps)

        app_metrics_response = cls(
            summary=summary,
            apps=apps,
        )

        app_metrics_response.additional_properties = d
        return app_metrics_response

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
