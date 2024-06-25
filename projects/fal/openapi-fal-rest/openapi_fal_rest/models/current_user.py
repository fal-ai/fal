from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..models.lock_reason import LockReason
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.user_member import UserMember


T = TypeVar("T", bound="CurrentUser")


@attr.s(auto_attribs=True)
class CurrentUser:
    """
    Attributes:
        full_name (str):
        nickname (str):
        user_id (str):
        is_personal (bool):
        is_locked (bool):
        lock_reason (Union[Unset, LockReason]): An enumeration.
        members (Union[Unset, List['UserMember']]):
        is_paying (Union[Unset, bool]):
    """

    full_name: str
    nickname: str
    user_id: str
    is_personal: bool
    is_locked: bool
    lock_reason: Union[Unset, LockReason] = UNSET
    members: Union[Unset, List["UserMember"]] = UNSET
    is_paying: Union[Unset, bool] = False
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        full_name = self.full_name
        nickname = self.nickname
        user_id = self.user_id
        is_personal = self.is_personal
        is_locked = self.is_locked
        lock_reason: Union[Unset, str] = UNSET
        if not isinstance(self.lock_reason, Unset):
            lock_reason = self.lock_reason.value

        members: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.members, Unset):
            members = []
            for members_item_data in self.members:
                members_item = members_item_data.to_dict()

                members.append(members_item)

        is_paying = self.is_paying

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "full_name": full_name,
                "nickname": nickname,
                "user_id": user_id,
                "is_personal": is_personal,
                "is_locked": is_locked,
            }
        )
        if lock_reason is not UNSET:
            field_dict["lock_reason"] = lock_reason
        if members is not UNSET:
            field_dict["members"] = members
        if is_paying is not UNSET:
            field_dict["is_paying"] = is_paying

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.user_member import UserMember

        d = src_dict.copy()
        full_name = d.pop("full_name")

        nickname = d.pop("nickname")

        user_id = d.pop("user_id")

        is_personal = d.pop("is_personal")

        is_locked = d.pop("is_locked")

        _lock_reason = d.pop("lock_reason", UNSET)
        lock_reason: Union[Unset, LockReason]
        if _lock_reason is None or isinstance(_lock_reason, Unset):
            lock_reason = UNSET
        else:
            lock_reason = LockReason(_lock_reason)

        members = []
        _members = d.pop("members", UNSET)
        for members_item_data in _members or []:
            members_item = UserMember.from_dict(members_item_data)

            members.append(members_item)

        is_paying = d.pop("is_paying", UNSET)

        current_user = cls(
            full_name=full_name,
            nickname=nickname,
            user_id=user_id,
            is_personal=is_personal,
            is_locked=is_locked,
            lock_reason=lock_reason,
            members=members,
            is_paying=is_paying,
        )

        current_user.additional_properties = d
        return current_user

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
