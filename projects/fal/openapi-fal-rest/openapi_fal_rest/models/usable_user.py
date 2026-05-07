import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.account_type import AccountType
from ..models.lock_reason import LockReason
from ..types import UNSET, Unset

T = TypeVar("T", bound="UsableUser")


@_attrs_define
class UsableUser:
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

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
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

        usable_user = cls(
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
        )

        usable_user.additional_properties = d
        return usable_user

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
