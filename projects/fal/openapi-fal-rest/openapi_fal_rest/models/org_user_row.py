from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.org_auth_user import OrgAuthUser
    from ..models.org_pending_invite_team import OrgPendingInviteTeam
    from ..models.org_user_team_assignment import OrgUserTeamAssignment


T = TypeVar("T", bound="OrgUserRow")


@_attrs_define
class OrgUserRow:
    """A row in the org users listing, grouped by email.

    Attributes:
        email (str):
        auth_users (list['OrgAuthUser']):
        team_assignments (list['OrgUserTeamAssignment']):
        pending_invites (Union[Unset, list['OrgPendingInviteTeam']]):
    """

    email: str
    auth_users: list["OrgAuthUser"]
    team_assignments: list["OrgUserTeamAssignment"]
    pending_invites: Union[Unset, list["OrgPendingInviteTeam"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        email = self.email

        auth_users = []
        for auth_users_item_data in self.auth_users:
            auth_users_item = auth_users_item_data.to_dict()
            auth_users.append(auth_users_item)

        team_assignments = []
        for team_assignments_item_data in self.team_assignments:
            team_assignments_item = team_assignments_item_data.to_dict()
            team_assignments.append(team_assignments_item)

        pending_invites: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.pending_invites, Unset):
            pending_invites = []
            for pending_invites_item_data in self.pending_invites:
                pending_invites_item = pending_invites_item_data.to_dict()
                pending_invites.append(pending_invites_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "email": email,
                "auth_users": auth_users,
                "team_assignments": team_assignments,
            }
        )
        if pending_invites is not UNSET:
            field_dict["pending_invites"] = pending_invites

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.org_auth_user import OrgAuthUser
        from ..models.org_pending_invite_team import OrgPendingInviteTeam
        from ..models.org_user_team_assignment import OrgUserTeamAssignment

        d = src_dict.copy()
        email = d.pop("email")

        auth_users = []
        _auth_users = d.pop("auth_users")
        for auth_users_item_data in _auth_users:
            auth_users_item = OrgAuthUser.from_dict(auth_users_item_data)

            auth_users.append(auth_users_item)

        team_assignments = []
        _team_assignments = d.pop("team_assignments")
        for team_assignments_item_data in _team_assignments:
            team_assignments_item = OrgUserTeamAssignment.from_dict(team_assignments_item_data)

            team_assignments.append(team_assignments_item)

        pending_invites = []
        _pending_invites = d.pop("pending_invites", UNSET)
        for pending_invites_item_data in _pending_invites or []:
            pending_invites_item = OrgPendingInviteTeam.from_dict(pending_invites_item_data)

            pending_invites.append(pending_invites_item)

        org_user_row = cls(
            email=email,
            auth_users=auth_users,
            team_assignments=team_assignments,
            pending_invites=pending_invites,
        )

        org_user_row.additional_properties = d
        return org_user_row

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
