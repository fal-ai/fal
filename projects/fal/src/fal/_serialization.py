from __future__ import annotations

from functools import wraps
from pathlib import Path

import dill
from dill import _dill

from fal.toolkit import mainify

# each @fal.function gets added to this set so that we can
# mainify the module this function is in
_MODULES: set[str] = set()
_PACKAGES: set[str] = set()


@mainify
def _pydantic_make_field(kwargs):
    from pydantic.fields import ModelField

    return ModelField(**kwargs)


@mainify
def _pydantic_make_private_field(kwargs):
    from pydantic.fields import ModelPrivateAttr

    return ModelPrivateAttr(**kwargs)


# this allows us to record all the "isolated" function and then mainify everything in
# module they exist
@wraps(_dill._locate_function)
def by_value_locator(obj, pickler=None, og_locator=_dill._locate_function):
    module_name = getattr(obj, "__module__", None)
    if module_name is not None:
        # If it is coming from the same module, directly allow
        # it to be pickled.
        if module_name in _MODULES:
            return False

        package_name, *_ = module_name.partition(".")
        # If it is coming from the same package, then do the same.
        if package_name in _PACKAGES:
            return False

    og_result = og_locator(obj, pickler)
    return og_result


_dill._locate_function = by_value_locator


def include_packages_from_path(raw_path: str):
    path = Path(raw_path).resolve()
    parent = path
    while (parent.parent / "__init__.py").exists():
        parent = parent.parent

    if parent != path:
        _PACKAGES.add(parent.name)


def add_serialization_listeners_for(obj):
    module_name = getattr(obj, "__module__", None)
    if not module_name:
        return None

    _MODULES.add(module_name)
    if module_name == "__main__":
        # When the module is __main__, we need to recursively go up the
        # tree to locate the actual package name.
        import __main__

        include_packages_from_path(__main__.__file__)

    if "." in module_name:
        package_name, *_ = module_name.partition(".")
        _PACKAGES.add(package_name)


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
