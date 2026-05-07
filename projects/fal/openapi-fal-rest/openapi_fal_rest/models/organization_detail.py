import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.account_type import AccountType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.org_config import OrgConfig
    from ..models.org_team_user import OrgTeamUser
    from ..models.team_member_info import TeamMemberInfo


T = TypeVar("T", bound="OrganizationDetail")


@_attrs_define
class OrganizationDetail:
    """Detailed information about an organization.

    Attributes:
        org_user_id (str):
        nickname (str):
        name (str):
        email (str):
        account_type (AccountType):
        created_at (datetime.datetime):
        teams (list['OrgTeamUser']):
        org_admins (Union[Unset, list['TeamMemberInfo']]):
        total_user_count (Union[Unset, int]):
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
        owned_sso_connections (Union[Unset, list[str]]):
    """

    org_user_id: str
    nickname: str
    name: str
    email: str
    account_type: AccountType
    created_at: datetime.datetime
    teams: list["OrgTeamUser"]
    org_admins: Union[Unset, list["TeamMemberInfo"]] = UNSET
    total_user_count: Union[Unset, int] = UNSET
    org_config: Union[Unset, "OrgConfig"] = UNSET
    owned_sso_connections: Union[Unset, list[str]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        org_user_id = self.org_user_id

        nickname = self.nickname

        name = self.name

        email = self.email

        account_type = self.account_type.value

        created_at = self.created_at.isoformat()

        teams = []
        for teams_item_data in self.teams:
            teams_item = teams_item_data.to_dict()
            teams.append(teams_item)

        org_admins: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.org_admins, Unset):
            org_admins = []
            for org_admins_item_data in self.org_admins:
                org_admins_item = org_admins_item_data.to_dict()
                org_admins.append(org_admins_item)

        total_user_count = self.total_user_count

        org_config: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.org_config, Unset):
            org_config = self.org_config.to_dict()

        owned_sso_connections: Union[Unset, list[str]] = UNSET
        if not isinstance(self.owned_sso_connections, Unset):
            owned_sso_connections = self.owned_sso_connections

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "org_user_id": org_user_id,
                "nickname": nickname,
                "name": name,
                "email": email,
                "account_type": account_type,
                "created_at": created_at,
                "teams": teams,
            }
        )
        if org_admins is not UNSET:
            field_dict["org_admins"] = org_admins
        if total_user_count is not UNSET:
            field_dict["total_user_count"] = total_user_count
        if org_config is not UNSET:
            field_dict["org_config"] = org_config
        if owned_sso_connections is not UNSET:
            field_dict["owned_sso_connections"] = owned_sso_connections

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.org_config import OrgConfig
        from ..models.org_team_user import OrgTeamUser
        from ..models.team_member_info import TeamMemberInfo

        d = src_dict.copy()
        org_user_id = d.pop("org_user_id")

        nickname = d.pop("nickname")

        name = d.pop("name")

        email = d.pop("email")

        account_type = AccountType(d.pop("account_type"))

        created_at = isoparse(d.pop("created_at"))

        teams = []
        _teams = d.pop("teams")
        for teams_item_data in _teams:
            teams_item = OrgTeamUser.from_dict(teams_item_data)

            teams.append(teams_item)

        org_admins = []
        _org_admins = d.pop("org_admins", UNSET)
        for org_admins_item_data in _org_admins or []:
            org_admins_item = TeamMemberInfo.from_dict(org_admins_item_data)

            org_admins.append(org_admins_item)

        total_user_count = d.pop("total_user_count", UNSET)

        _org_config = d.pop("org_config", UNSET)
        org_config: Union[Unset, OrgConfig]
        if isinstance(_org_config, Unset):
            org_config = UNSET
        else:
            org_config = OrgConfig.from_dict(_org_config)

        owned_sso_connections = cast(list[str], d.pop("owned_sso_connections", UNSET))

        organization_detail = cls(
            org_user_id=org_user_id,
            nickname=nickname,
            name=name,
            email=email,
            account_type=account_type,
            created_at=created_at,
            teams=teams,
            org_admins=org_admins,
            total_user_count=total_user_count,
            org_config=org_config,
            owned_sso_connections=owned_sso_connections,
        )

        organization_detail.additional_properties = d
        return organization_detail

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
