import datetime
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

if TYPE_CHECKING:
    from ..models.order_form_terms import OrderFormTerms


T = TypeVar("T", bound="OrderForm")


@_attrs_define
class OrderForm:
    """
    Attributes:
        order_form_id (UUID):
        user_id (str):
        created_by (str):
        created_at (datetime.datetime):
        term_start (datetime.datetime):
        term_end (datetime.datetime):
        terms (OrderFormTerms):
    """

    order_form_id: UUID
    user_id: str
    created_by: str
    created_at: datetime.datetime
    term_start: datetime.datetime
    term_end: datetime.datetime
    terms: "OrderFormTerms"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        order_form_id = str(self.order_form_id)

        user_id = self.user_id

        created_by = self.created_by

        created_at = self.created_at.isoformat()

        term_start = self.term_start.isoformat()

        term_end = self.term_end.isoformat()

        terms = self.terms.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "order_form_id": order_form_id,
                "user_id": user_id,
                "created_by": created_by,
                "created_at": created_at,
                "term_start": term_start,
                "term_end": term_end,
                "terms": terms,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.order_form_terms import OrderFormTerms

        d = src_dict.copy()
        order_form_id = UUID(d.pop("order_form_id"))

        user_id = d.pop("user_id")

        created_by = d.pop("created_by")

        created_at = isoparse(d.pop("created_at"))

        term_start = isoparse(d.pop("term_start"))

        term_end = isoparse(d.pop("term_end"))

        terms = OrderFormTerms.from_dict(d.pop("terms"))

        order_form = cls(
            order_form_id=order_form_id,
            user_id=user_id,
            created_by=created_by,
            created_at=created_at,
            term_start=term_start,
            term_end=term_end,
            terms=terms,
        )

        order_form.additional_properties = d
        return order_form

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
