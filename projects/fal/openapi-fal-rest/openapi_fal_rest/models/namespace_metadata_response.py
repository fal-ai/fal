from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.namespace_index_info import NamespaceIndexInfo
    from ..models.namespace_metadata_response_encryption import NamespaceMetadataResponseEncryption
    from ..models.namespace_metadata_response_schema import NamespaceMetadataResponseSchema


T = TypeVar("T", bound="NamespaceMetadataResponse")


@_attrs_define
class NamespaceMetadataResponse:
    """
    Attributes:
        schema (Union[Unset, NamespaceMetadataResponseSchema]):
        approx_logical_bytes (Union[Unset, int]):
        approx_row_count (Union[Unset, int]):
        created_at (Union[Unset, str]):
        updated_at (Union[Unset, str]):
        encryption (Union[Unset, NamespaceMetadataResponseEncryption]):
        index (Union[Unset, NamespaceIndexInfo]):
    """

    schema: Union[Unset, "NamespaceMetadataResponseSchema"] = UNSET
    approx_logical_bytes: Union[Unset, int] = UNSET
    approx_row_count: Union[Unset, int] = UNSET
    created_at: Union[Unset, str] = UNSET
    updated_at: Union[Unset, str] = UNSET
    encryption: Union[Unset, "NamespaceMetadataResponseEncryption"] = UNSET
    index: Union[Unset, "NamespaceIndexInfo"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        schema: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.schema, Unset):
            schema = self.schema.to_dict()

        approx_logical_bytes = self.approx_logical_bytes

        approx_row_count = self.approx_row_count

        created_at = self.created_at

        updated_at = self.updated_at

        encryption: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.encryption, Unset):
            encryption = self.encryption.to_dict()

        index: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.index, Unset):
            index = self.index.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if schema is not UNSET:
            field_dict["schema"] = schema
        if approx_logical_bytes is not UNSET:
            field_dict["approx_logical_bytes"] = approx_logical_bytes
        if approx_row_count is not UNSET:
            field_dict["approx_row_count"] = approx_row_count
        if created_at is not UNSET:
            field_dict["created_at"] = created_at
        if updated_at is not UNSET:
            field_dict["updated_at"] = updated_at
        if encryption is not UNSET:
            field_dict["encryption"] = encryption
        if index is not UNSET:
            field_dict["index"] = index

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.namespace_index_info import NamespaceIndexInfo
        from ..models.namespace_metadata_response_encryption import NamespaceMetadataResponseEncryption
        from ..models.namespace_metadata_response_schema import NamespaceMetadataResponseSchema

        d = src_dict.copy()
        _schema = d.pop("schema", UNSET)
        schema: Union[Unset, NamespaceMetadataResponseSchema]
        if isinstance(_schema, Unset):
            schema = UNSET
        else:
            schema = NamespaceMetadataResponseSchema.from_dict(_schema)

        approx_logical_bytes = d.pop("approx_logical_bytes", UNSET)

        approx_row_count = d.pop("approx_row_count", UNSET)

        created_at = d.pop("created_at", UNSET)

        updated_at = d.pop("updated_at", UNSET)

        _encryption = d.pop("encryption", UNSET)
        encryption: Union[Unset, NamespaceMetadataResponseEncryption]
        if isinstance(_encryption, Unset):
            encryption = UNSET
        else:
            encryption = NamespaceMetadataResponseEncryption.from_dict(_encryption)

        _index = d.pop("index", UNSET)
        index: Union[Unset, NamespaceIndexInfo]
        if isinstance(_index, Unset):
            index = UNSET
        else:
            index = NamespaceIndexInfo.from_dict(_index)

        namespace_metadata_response = cls(
            schema=schema,
            approx_logical_bytes=approx_logical_bytes,
            approx_row_count=approx_row_count,
            created_at=created_at,
            updated_at=updated_at,
            encryption=encryption,
            index=index,
        )

        namespace_metadata_response.additional_properties = d
        return namespace_metadata_response

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
