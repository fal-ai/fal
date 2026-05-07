from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.account_type import AccountType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.admin_config import AdminConfig
    from ..models.object_lifecycle_preference import ObjectLifecyclePreference


T = TypeVar("T", bound="UpdatableUserInfo")


@_attrs_define
class UpdatableUserInfo:
    """
    Attributes:
        user_id (str):
        email (str):
        nickname (str):
        is_personal (bool):
        full_name (str):
        allow_request_io_storage (Union[Unset, bool]):  Default: True.
        auto_control_auth_provider (Union[Unset, str]):
        admin_config (Union[Unset, AdminConfig]):
        object_lifecycle_preference (Union[Unset, ObjectLifecyclePreference]):
        salesforce_id (Union[Unset, str]):
        account_type (Union[Unset, AccountType]):
    """

    user_id: str
    email: str
    nickname: str
    is_personal: bool
    full_name: str
    allow_request_io_storage: Union[Unset, bool] = True
    auto_control_auth_provider: Union[Unset, str] = UNSET
    admin_config: Union[Unset, "AdminConfig"] = UNSET
    object_lifecycle_preference: Union[Unset, "ObjectLifecyclePreference"] = UNSET
    salesforce_id: Union[Unset, str] = UNSET
    account_type: Union[Unset, AccountType] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        email = self.email

        nickname = self.nickname

        is_personal = self.is_personal

        full_name = self.full_name

        allow_request_io_storage = self.allow_request_io_storage

        auto_control_auth_provider = self.auto_control_auth_provider

        admin_config: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.admin_config, Unset):
            admin_config = self.admin_config.to_dict()

        object_lifecycle_preference: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.object_lifecycle_preference, Unset):
            object_lifecycle_preference = self.object_lifecycle_preference.to_dict()

        salesforce_id = self.salesforce_id

        account_type: Union[Unset, str] = UNSET
        if not isinstance(self.account_type, Unset):
            account_type = self.account_type.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "email": email,
                "nickname": nickname,
                "is_personal": is_personal,
                "full_name": full_name,
            }
        )
        if allow_request_io_storage is not UNSET:
            field_dict["allow_request_io_storage"] = allow_request_io_storage
        if auto_control_auth_provider is not UNSET:
            field_dict["auto_control_auth_provider"] = auto_control_auth_provider
        if admin_config is not UNSET:
            field_dict["admin_config"] = admin_config
        if object_lifecycle_preference is not UNSET:
            field_dict["object_lifecycle_preference"] = object_lifecycle_preference
        if salesforce_id is not UNSET:
            field_dict["salesforce_id"] = salesforce_id
        if account_type is not UNSET:
            field_dict["account_type"] = account_type

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.admin_config import AdminConfig
        from ..models.object_lifecycle_preference import ObjectLifecyclePreference

        d = src_dict.copy()
        user_id = d.pop("user_id")

        email = d.pop("email")

        nickname = d.pop("nickname")

        is_personal = d.pop("is_personal")

        full_name = d.pop("full_name")

        allow_request_io_storage = d.pop("allow_request_io_storage", UNSET)

        auto_control_auth_provider = d.pop("auto_control_auth_provider", UNSET)

        _admin_config = d.pop("admin_config", UNSET)
        admin_config: Union[Unset, AdminConfig]
        if isinstance(_admin_config, Unset):
            admin_config = UNSET
        else:
            admin_config = AdminConfig.from_dict(_admin_config)

        _object_lifecycle_preference = d.pop("object_lifecycle_preference", UNSET)
        object_lifecycle_preference: Union[Unset, ObjectLifecyclePreference]
        if isinstance(_object_lifecycle_preference, Unset):
            object_lifecycle_preference = UNSET
        else:
            object_lifecycle_preference = ObjectLifecyclePreference.from_dict(_object_lifecycle_preference)

        salesforce_id = d.pop("salesforce_id", UNSET)

        _account_type = d.pop("account_type", UNSET)
        account_type: Union[Unset, AccountType]
        if isinstance(_account_type, Unset):
            account_type = UNSET
        else:
            account_type = AccountType(_account_type)

        updatable_user_info = cls(
            user_id=user_id,
            email=email,
            nickname=nickname,
            is_personal=is_personal,
            full_name=full_name,
            allow_request_io_storage=allow_request_io_storage,
            auto_control_auth_provider=auto_control_auth_provider,
            admin_config=admin_config,
            object_lifecycle_preference=object_lifecycle_preference,
            salesforce_id=salesforce_id,
            account_type=account_type,
        )

        updatable_user_info.additional_properties = d
        return updatable_user_info

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
