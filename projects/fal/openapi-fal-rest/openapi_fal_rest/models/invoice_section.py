from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.api_key_subsection import ApiKeySubsection
    from ..models.playground_subsection import PlaygroundSubsection
    from ..models.sub_item import SubItem


T = TypeVar("T", bound="InvoiceSection")


@_attrs_define
class InvoiceSection:
    """A section of the invoice (e.g., Endpoint Output, Compute Seconds).

    Attributes:
        name (str):
        amount (Union[Unset, str]):
        subtotal (Union[Unset, str]):
        quantity (Union[Unset, float]):
        start_date (Union[Unset, str]):
        end_date (Union[Unset, str]):
        subsections (Union[Unset, list[Union['ApiKeySubsection', 'PlaygroundSubsection']]]):
        sub_items (Union[Unset, list['SubItem']]):
    """

    name: str
    amount: Union[Unset, str] = UNSET
    subtotal: Union[Unset, str] = UNSET
    quantity: Union[Unset, float] = UNSET
    start_date: Union[Unset, str] = UNSET
    end_date: Union[Unset, str] = UNSET
    subsections: Union[Unset, list[Union["ApiKeySubsection", "PlaygroundSubsection"]]] = UNSET
    sub_items: Union[Unset, list["SubItem"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.api_key_subsection import ApiKeySubsection

        name = self.name

        amount = self.amount

        subtotal = self.subtotal

        quantity = self.quantity

        start_date = self.start_date

        end_date = self.end_date

        subsections: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.subsections, Unset):
            subsections = []
            for subsections_item_data in self.subsections:
                subsections_item: dict[str, Any]
                if isinstance(subsections_item_data, ApiKeySubsection):
                    subsections_item = subsections_item_data.to_dict()
                else:
                    subsections_item = subsections_item_data.to_dict()

                subsections.append(subsections_item)

        sub_items: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.sub_items, Unset):
            sub_items = []
            for sub_items_item_data in self.sub_items:
                sub_items_item = sub_items_item_data.to_dict()
                sub_items.append(sub_items_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
            }
        )
        if amount is not UNSET:
            field_dict["amount"] = amount
        if subtotal is not UNSET:
            field_dict["subtotal"] = subtotal
        if quantity is not UNSET:
            field_dict["quantity"] = quantity
        if start_date is not UNSET:
            field_dict["start_date"] = start_date
        if end_date is not UNSET:
            field_dict["end_date"] = end_date
        if subsections is not UNSET:
            field_dict["subsections"] = subsections
        if sub_items is not UNSET:
            field_dict["sub_items"] = sub_items

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.api_key_subsection import ApiKeySubsection
        from ..models.playground_subsection import PlaygroundSubsection
        from ..models.sub_item import SubItem

        d = src_dict.copy()
        name = d.pop("name")

        amount = d.pop("amount", UNSET)

        subtotal = d.pop("subtotal", UNSET)

        quantity = d.pop("quantity", UNSET)

        start_date = d.pop("start_date", UNSET)

        end_date = d.pop("end_date", UNSET)

        subsections = []
        _subsections = d.pop("subsections", UNSET)
        for subsections_item_data in _subsections or []:

            def _parse_subsections_item(data: object) -> Union["ApiKeySubsection", "PlaygroundSubsection"]:
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    subsections_item_type_0 = ApiKeySubsection.from_dict(data)

                    return subsections_item_type_0
                except:  # noqa: E722
                    pass
                if not isinstance(data, dict):
                    raise TypeError()
                subsections_item_type_1 = PlaygroundSubsection.from_dict(data)

                return subsections_item_type_1

            subsections_item = _parse_subsections_item(subsections_item_data)

            subsections.append(subsections_item)

        sub_items = []
        _sub_items = d.pop("sub_items", UNSET)
        for sub_items_item_data in _sub_items or []:
            sub_items_item = SubItem.from_dict(sub_items_item_data)

            sub_items.append(sub_items_item)

        invoice_section = cls(
            name=name,
            amount=amount,
            subtotal=subtotal,
            quantity=quantity,
            start_date=start_date,
            end_date=end_date,
            subsections=subsections,
            sub_items=sub_items,
        )

        invoice_section.additional_properties = d
        return invoice_section

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
