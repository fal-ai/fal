from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.org_config import OrgConfig


T = TypeVar("T", bound="CreateOrganizationRequest")


@_attrs_define
class CreateOrganizationRequest:
    """Request to create a new organization.

    Attributes:
        name (str):
        email (str):
        admin_user_str (Union[Unset, str]):
        auto_control_auth_provider (Union[Unset, str]):
        org_config (Union[Unset, OrgConfig]): Org-specific configuration derived from org admin config.

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
                org_user_restricted_request_view: Restrict users to only view their own requests (org and child teams).
            Default: False.
                owned_sso_connections: List of SSO connections owned by the org.
                    Used when creating invites to specify which SSO connection is required.
                org_endpoint_access_controls: Whether the org has endpoint access controls enabled (for inheritance).
            Default: False.
                org_external_app_sharing: Allow sharing apps with users outside the org via per-user auth overrides.
                    When False, per-user auth overrides can only target users within the organization. Default: True.
                is_early_access: Whether the org is enrolled in the
                    Early Access Program. When True, early access models
                    are surfaced in Model Access Controls. Default: False.
    """

    name: str
    email: str
    admin_user_str: Union[Unset, str] = UNSET
    auto_control_auth_provider: Union[Unset, str] = UNSET
    org_config: Union[Unset, "OrgConfig"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        email = self.email

        admin_user_str = self.admin_user_str

        auto_control_auth_provider = self.auto_control_auth_provider

        org_config: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.org_config, Unset):
            org_config = self.org_config.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "email": email,
            }
        )
        if admin_user_str is not UNSET:
            field_dict["admin_user_str"] = admin_user_str
        if auto_control_auth_provider is not UNSET:
            field_dict["auto_control_auth_provider"] = auto_control_auth_provider
        if org_config is not UNSET:
            field_dict["org_config"] = org_config

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.org_config import OrgConfig

        d = src_dict.copy()
        name = d.pop("name")

        email = d.pop("email")

        admin_user_str = d.pop("admin_user_str", UNSET)

        auto_control_auth_provider = d.pop("auto_control_auth_provider", UNSET)

        _org_config = d.pop("org_config", UNSET)
        org_config: Union[Unset, OrgConfig]
        if isinstance(_org_config, Unset):
            org_config = UNSET
        else:
            org_config = OrgConfig.from_dict(_org_config)

        create_organization_request = cls(
            name=name,
            email=email,
            admin_user_str=admin_user_str,
            auto_control_auth_provider=auto_control_auth_provider,
            org_config=org_config,
        )

        create_organization_request.additional_properties = d
        return create_organization_request

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
