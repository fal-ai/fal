import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.account_type import AccountType
from ..models.admin_role import AdminRole
from ..models.lock_reason import LockReason
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.extra_permissions import ExtraPermissions
    from ..models.org_billing import OrgBilling
    from ..models.org_config import OrgConfig
    from ..models.usable_user import UsableUser
    from ..models.user_member import UserMember


T = TypeVar("T", bound="CurrentUser")


@_attrs_define
class CurrentUser:
    """
    Attributes:
        full_name (str):
        nickname (str):
        email (str):
        user_id (str):
        is_personal (bool):
        is_locked (bool):
        account_type (AccountType):
        created_at (datetime.datetime):
        lock_reason (Union[Unset, LockReason]):
        org_user_id (Union[Unset, str]):
        org_name (Union[Unset, str]):
        is_org (Union[Unset, bool]):  Default: False.
        members (Union[Unset, list['UserMember']]):
        is_invoicing (Union[Unset, bool]):  Default: False.
        extra_permissions (Union[Unset, ExtraPermissions]):
        teams (Union[Unset, list['UsableUser']]):
        concurrent_requests_limit (Union[Unset, int]):
        is_gcp_marketplace (Union[Unset, bool]):  Default: False.
        orb_customer_id (Union[Unset, str]):
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
        org_billing (Union[Unset, OrgBilling]): Org-wide billing summary for org admins/billing users.

            Attributes:
                has_invoicing: Whether any team in the org uses invoicing.
        sso_personal_account_hidden (Union[Unset, bool]):  Default: False.
        admin_roles (Union[Unset, list[AdminRole]]):
    """

    full_name: str
    nickname: str
    email: str
    user_id: str
    is_personal: bool
    is_locked: bool
    account_type: AccountType
    created_at: datetime.datetime
    lock_reason: Union[Unset, LockReason] = UNSET
    org_user_id: Union[Unset, str] = UNSET
    org_name: Union[Unset, str] = UNSET
    is_org: Union[Unset, bool] = False
    members: Union[Unset, list["UserMember"]] = UNSET
    is_invoicing: Union[Unset, bool] = False
    extra_permissions: Union[Unset, "ExtraPermissions"] = UNSET
    teams: Union[Unset, list["UsableUser"]] = UNSET
    concurrent_requests_limit: Union[Unset, int] = UNSET
    is_gcp_marketplace: Union[Unset, bool] = False
    orb_customer_id: Union[Unset, str] = UNSET
    org_config: Union[Unset, "OrgConfig"] = UNSET
    org_billing: Union[Unset, "OrgBilling"] = UNSET
    sso_personal_account_hidden: Union[Unset, bool] = False
    admin_roles: Union[Unset, list[AdminRole]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        full_name = self.full_name

        nickname = self.nickname

        email = self.email

        user_id = self.user_id

        is_personal = self.is_personal

        is_locked = self.is_locked

        account_type = self.account_type.value

        created_at = self.created_at.isoformat()

        lock_reason: Union[Unset, str] = UNSET
        if not isinstance(self.lock_reason, Unset):
            lock_reason = self.lock_reason.value

        org_user_id = self.org_user_id

        org_name = self.org_name

        is_org = self.is_org

        members: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.members, Unset):
            members = []
            for members_item_data in self.members:
                members_item = members_item_data.to_dict()
                members.append(members_item)

        is_invoicing = self.is_invoicing

        extra_permissions: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.extra_permissions, Unset):
            extra_permissions = self.extra_permissions.to_dict()

        teams: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.teams, Unset):
            teams = []
            for teams_item_data in self.teams:
                teams_item = teams_item_data.to_dict()
                teams.append(teams_item)

        concurrent_requests_limit = self.concurrent_requests_limit

        is_gcp_marketplace = self.is_gcp_marketplace

        orb_customer_id = self.orb_customer_id

        org_config: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.org_config, Unset):
            org_config = self.org_config.to_dict()

        org_billing: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.org_billing, Unset):
            org_billing = self.org_billing.to_dict()

        sso_personal_account_hidden = self.sso_personal_account_hidden

        admin_roles: Union[Unset, list[str]] = UNSET
        if not isinstance(self.admin_roles, Unset):
            admin_roles = []
            for admin_roles_item_data in self.admin_roles:
                admin_roles_item = admin_roles_item_data.value
                admin_roles.append(admin_roles_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "full_name": full_name,
                "nickname": nickname,
                "email": email,
                "user_id": user_id,
                "is_personal": is_personal,
                "is_locked": is_locked,
                "account_type": account_type,
                "created_at": created_at,
            }
        )
        if lock_reason is not UNSET:
            field_dict["lock_reason"] = lock_reason
        if org_user_id is not UNSET:
            field_dict["org_user_id"] = org_user_id
        if org_name is not UNSET:
            field_dict["org_name"] = org_name
        if is_org is not UNSET:
            field_dict["is_org"] = is_org
        if members is not UNSET:
            field_dict["members"] = members
        if is_invoicing is not UNSET:
            field_dict["is_invoicing"] = is_invoicing
        if extra_permissions is not UNSET:
            field_dict["extra_permissions"] = extra_permissions
        if teams is not UNSET:
            field_dict["teams"] = teams
        if concurrent_requests_limit is not UNSET:
            field_dict["concurrent_requests_limit"] = concurrent_requests_limit
        if is_gcp_marketplace is not UNSET:
            field_dict["is_gcp_marketplace"] = is_gcp_marketplace
        if orb_customer_id is not UNSET:
            field_dict["orb_customer_id"] = orb_customer_id
        if org_config is not UNSET:
            field_dict["org_config"] = org_config
        if org_billing is not UNSET:
            field_dict["org_billing"] = org_billing
        if sso_personal_account_hidden is not UNSET:
            field_dict["sso_personal_account_hidden"] = sso_personal_account_hidden
        if admin_roles is not UNSET:
            field_dict["admin_roles"] = admin_roles

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.extra_permissions import ExtraPermissions
        from ..models.org_billing import OrgBilling
        from ..models.org_config import OrgConfig
        from ..models.usable_user import UsableUser
        from ..models.user_member import UserMember

        d = src_dict.copy()
        full_name = d.pop("full_name")

        nickname = d.pop("nickname")

        email = d.pop("email")

        user_id = d.pop("user_id")

        is_personal = d.pop("is_personal")

        is_locked = d.pop("is_locked")

        account_type = AccountType(d.pop("account_type"))

        created_at = isoparse(d.pop("created_at"))

        _lock_reason = d.pop("lock_reason", UNSET)
        lock_reason: Union[Unset, LockReason]
        if isinstance(_lock_reason, Unset):
            lock_reason = UNSET
        else:
            lock_reason = LockReason(_lock_reason)

        org_user_id = d.pop("org_user_id", UNSET)

        org_name = d.pop("org_name", UNSET)

        is_org = d.pop("is_org", UNSET)

        members = []
        _members = d.pop("members", UNSET)
        for members_item_data in _members or []:
            members_item = UserMember.from_dict(members_item_data)

            members.append(members_item)

        is_invoicing = d.pop("is_invoicing", UNSET)

        _extra_permissions = d.pop("extra_permissions", UNSET)
        extra_permissions: Union[Unset, ExtraPermissions]
        if isinstance(_extra_permissions, Unset):
            extra_permissions = UNSET
        else:
            extra_permissions = ExtraPermissions.from_dict(_extra_permissions)

        teams = []
        _teams = d.pop("teams", UNSET)
        for teams_item_data in _teams or []:
            teams_item = UsableUser.from_dict(teams_item_data)

            teams.append(teams_item)

        concurrent_requests_limit = d.pop("concurrent_requests_limit", UNSET)

        is_gcp_marketplace = d.pop("is_gcp_marketplace", UNSET)

        orb_customer_id = d.pop("orb_customer_id", UNSET)

        _org_config = d.pop("org_config", UNSET)
        org_config: Union[Unset, OrgConfig]
        if isinstance(_org_config, Unset):
            org_config = UNSET
        else:
            org_config = OrgConfig.from_dict(_org_config)

        _org_billing = d.pop("org_billing", UNSET)
        org_billing: Union[Unset, OrgBilling]
        if isinstance(_org_billing, Unset):
            org_billing = UNSET
        else:
            org_billing = OrgBilling.from_dict(_org_billing)

        sso_personal_account_hidden = d.pop("sso_personal_account_hidden", UNSET)

        admin_roles = []
        _admin_roles = d.pop("admin_roles", UNSET)
        for admin_roles_item_data in _admin_roles or []:
            admin_roles_item = AdminRole(admin_roles_item_data)

            admin_roles.append(admin_roles_item)

        current_user = cls(
            full_name=full_name,
            nickname=nickname,
            email=email,
            user_id=user_id,
            is_personal=is_personal,
            is_locked=is_locked,
            account_type=account_type,
            created_at=created_at,
            lock_reason=lock_reason,
            org_user_id=org_user_id,
            org_name=org_name,
            is_org=is_org,
            members=members,
            is_invoicing=is_invoicing,
            extra_permissions=extra_permissions,
            teams=teams,
            concurrent_requests_limit=concurrent_requests_limit,
            is_gcp_marketplace=is_gcp_marketplace,
            orb_customer_id=orb_customer_id,
            org_config=org_config,
            org_billing=org_billing,
            sso_personal_account_hidden=sso_personal_account_hidden,
            admin_roles=admin_roles,
        )

        current_user.additional_properties = d
        return current_user

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
