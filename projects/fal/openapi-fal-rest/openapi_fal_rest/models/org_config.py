from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="OrgConfig")


@_attrs_define
class OrgConfig:
    """Org-specific configuration derived from org admin config.

    Used for both input (creation/update) and output (resolved config).
    For input, None means "use default". For output, all fields are populated.

    Attributes:
        org_wide_views: Allow viewing across teams and org (org team only). Default: False.
        org_admin_view: Allow org admin view access (org team only). Default: False.
        org_team_lifecycle: Allow child teams to create and archive teams.
            When False, only org admins can create/archive teams; child team admins
            cannot self-service team creation or archive their own teams. Default: True.
        org_team_invites: Allow child teams to send invites.
            When False, only org admins can invite users to teams; child team admins
            cannot send invites from the team-level invite endpoints. Default: True.
        org_user_restricted_request_view: Restrict users to only view their own requests (org and child teams). Default:
    False.
        owned_sso_connections: List of SSO connections owned by the org.
            Used when creating invites to specify which SSO connection is required.
        org_endpoint_access_controls: Whether the org has endpoint access controls enabled (for inheritance). Default:
    False.
        org_external_app_sharing: Allow sharing apps with users outside the org via per-user auth overrides.
            When False, per-user auth overrides can only target users within the organization. Default: True.
        is_early_access: Whether the org is enrolled in the
            Early Access Program. When True, early access models
            are surfaced in Model Access Controls. Default: False.

        Attributes:
            org_wide_views (Union[Unset, bool]):
            org_admin_view (Union[Unset, bool]):
            org_team_lifecycle (Union[Unset, bool]):
            org_team_invites (Union[Unset, bool]):
            org_user_restricted_request_view (Union[Unset, bool]):
            owned_sso_connections (Union[Unset, list[str]]):
            org_endpoint_access_controls (Union[Unset, bool]):
            org_external_app_sharing (Union[Unset, bool]):
            is_early_access (Union[Unset, bool]):
    """

    org_wide_views: Union[Unset, bool] = UNSET
    org_admin_view: Union[Unset, bool] = UNSET
    org_team_lifecycle: Union[Unset, bool] = UNSET
    org_team_invites: Union[Unset, bool] = UNSET
    org_user_restricted_request_view: Union[Unset, bool] = UNSET
    owned_sso_connections: Union[Unset, list[str]] = UNSET
    org_endpoint_access_controls: Union[Unset, bool] = UNSET
    org_external_app_sharing: Union[Unset, bool] = UNSET
    is_early_access: Union[Unset, bool] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        org_wide_views = self.org_wide_views

        org_admin_view = self.org_admin_view

        org_team_lifecycle = self.org_team_lifecycle

        org_team_invites = self.org_team_invites

        org_user_restricted_request_view = self.org_user_restricted_request_view

        owned_sso_connections: Union[Unset, list[str]] = UNSET
        if not isinstance(self.owned_sso_connections, Unset):
            owned_sso_connections = self.owned_sso_connections

        org_endpoint_access_controls = self.org_endpoint_access_controls

        org_external_app_sharing = self.org_external_app_sharing

        is_early_access = self.is_early_access

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if org_wide_views is not UNSET:
            field_dict["org_wide_views"] = org_wide_views
        if org_admin_view is not UNSET:
            field_dict["org_admin_view"] = org_admin_view
        if org_team_lifecycle is not UNSET:
            field_dict["org_team_lifecycle"] = org_team_lifecycle
        if org_team_invites is not UNSET:
            field_dict["org_team_invites"] = org_team_invites
        if org_user_restricted_request_view is not UNSET:
            field_dict["org_user_restricted_request_view"] = org_user_restricted_request_view
        if owned_sso_connections is not UNSET:
            field_dict["owned_sso_connections"] = owned_sso_connections
        if org_endpoint_access_controls is not UNSET:
            field_dict["org_endpoint_access_controls"] = org_endpoint_access_controls
        if org_external_app_sharing is not UNSET:
            field_dict["org_external_app_sharing"] = org_external_app_sharing
        if is_early_access is not UNSET:
            field_dict["is_early_access"] = is_early_access

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        org_wide_views = d.pop("org_wide_views", UNSET)

        org_admin_view = d.pop("org_admin_view", UNSET)

        org_team_lifecycle = d.pop("org_team_lifecycle", UNSET)

        org_team_invites = d.pop("org_team_invites", UNSET)

        org_user_restricted_request_view = d.pop("org_user_restricted_request_view", UNSET)

        owned_sso_connections = cast(list[str], d.pop("owned_sso_connections", UNSET))

        org_endpoint_access_controls = d.pop("org_endpoint_access_controls", UNSET)

        org_external_app_sharing = d.pop("org_external_app_sharing", UNSET)

        is_early_access = d.pop("is_early_access", UNSET)

        org_config = cls(
            org_wide_views=org_wide_views,
            org_admin_view=org_admin_view,
            org_team_lifecycle=org_team_lifecycle,
            org_team_invites=org_team_invites,
            org_user_restricted_request_view=org_user_restricted_request_view,
            owned_sso_connections=owned_sso_connections,
            org_endpoint_access_controls=org_endpoint_access_controls,
            org_external_app_sharing=org_external_app_sharing,
            is_early_access=is_early_access,
        )

        org_config.additional_properties = d
        return org_config

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
