from __future__ import annotations

from fal.toolkit import mainify


@mainify
def _pydantic_make_field(kwargs):
    from pydantic.fields import ModelField

    return ModelField(**kwargs)


@mainify
def _pydantic_make_private_field(kwargs):
    from pydantic.fields import ModelPrivateAttr

    return ModelPrivateAttr(**kwargs)


@mainify
def patch_pydantic_field_serialization():
    # Cythonized pydantic fields can't be serialized automatically, so we are
    # have a special case handling for them that unpacks it to a dictionary
    # and then reloads it on the other side.
    import dill

    try:
        import pydantic.fields
    except ImportError:
        return

    @dill.register(pydantic.fields.ModelField)
    def _pickle_model_field(
        pickler: dill.Pickler,
        field: pydantic.fields.ModelField,
    ) -> None:
        args = {
            "name": field.name,
            # outer_type_ is the original type for ModelFields,
            # while type_ can be updated later with the nested type
            # like int for List[int].
            "type_": field.outer_type_,
            "class_validators": field.class_validators,
            "model_config": field.model_config,
            "default": field.default,
            "default_factory": field.default_factory,
            "required": field.required,
            "alias": field.alias,
            "field_info": field.field_info,
        }
        pickler.save_reduce(_pydantic_make_field, (args,), obj=field)

    @dill.register(pydantic.fields.ModelPrivateAttr)
    def _pickle_model_private_attr(
        pickler: dill.Pickler,
        field: pydantic.fields.ModelPrivateAttr,
    ) -> None:
        args = {
            "default": field.default,
            "default_factory": field.default_factory,
        }
        pickler.save_reduce(_pydantic_make_private_field, (args,), obj=field)


@mainify
def patch_pydantic_class_attributes():
    # Dill attempts to modify the __class__ of deserialized pydantic objects
    # on this side but it meets with a rejection from pydantic's semantics since
    # __class__ is not recognized as a proper dunder attribute.
    try:
        import pydantic.utils
    except ImportError:
        return

    pydantic.utils.DUNDER_ATTRIBUTES.add("__class__")


@mainify
def patch_dill():
    import dill

    dill.settings["recurse"] = True
    patch_pydantic_class_attributes()
    patch_pydantic_field_serialization()
