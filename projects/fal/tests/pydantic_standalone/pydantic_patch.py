"""A standalone version of the Pydantic v2 patch shipped in `fal._pydantic_patch`."""
from pprint import pprint
from typing import Callable, Optional, TypeVar, Union

import dill
import dill._dill as dill_serialization
import pydantic
from pydantic import BaseModel
from pydantic._internal._decorators import (
    Decorator,
    FieldValidatorDecoratorInfo,
    ModelValidatorDecoratorInfo,
)
from pydantic.config import ConfigDict
from pydantic.fields import FieldInfo

__all__ = [
    "build_pydantic_model",
    "extract_validators",
    "pickler_building_args",
    "_dill_hook_for_pydantic_models",
    "deserialise_pydantic_model",
]

ModelT = TypeVar("ModelT", bound=BaseModel)
"""An instance of any subclass of `pydantic.BaseModel`."""
Fields = dict[str, FieldInfo]
"""A mapping of field name to `pydantic.fields.FieldInfo`."""
ModelVs = dict[str, tuple[Callable, ModelValidatorDecoratorInfo]]
"""A mapping of model validator names to a tuple of the validator funcdef and info."""
FieldVs = dict[str, tuple[Callable, FieldValidatorDecoratorInfo]]
"""A mapping of field validator names to a tuple of the validator funcdef and info."""
Methods = dict[str, Callable]
"""A mapping of method names to the method funcdef."""
ModelVDeco = Decorator[ModelValidatorDecoratorInfo]
"""Alias for the decorator storing info about a model validator."""
FieldVDeco = Decorator[FieldValidatorDecoratorInfo]
"""Alias for the decorator storing info about a field validator."""


def build_pydantic_model(
    name: str,
    model_config: ConfigDict,
    model_doc: Optional[str],
    base_cls: type[ModelT],
    model_module: str,
    model_fields: Fields,
    model_validators: ModelVs,
    field_validators: FieldVs,
    methods: Methods,
) -> type[ModelT]:
    """Recreate the Pydantic model from the deserialised validator info.

    Arguments:
        name: The name of the model.
        model_config: The model's configuration settings. (UNUSED)
        model_doc: The model's docstring.
        base_cls: The model's base class.
        model_module: The name of the module the model belongs to.
        model_fields: The model's fields.
        model_validators: The model validators of the model.
        field_validators: The field validators of the model.
        methods: The methods of the model.
    """
    import pydantic

    validators = {
        **{
            name: pydantic.model_validator(mode=info.mode)(func)
            for name, (func, info) in model_validators.items()
        },
        **{
            name: pydantic.field_validator(*info.fields, mode=info.mode)(func)
            for name, (func, info) in field_validators.items()
        },
    }
    # TODO: review if this is the optimal way to handle model field annotations
    model_fields["__annotations__"] = {
        name: getattr(field, "annotation", None) for name, field in model_fields.items()
    }
    model_fields.update({name: func for name, (func, _) in field_validators.items()})

    model_cls = pydantic.create_model(
        name,
        # __config__=model_config, # UNUSED
        __doc__=model_doc,
        __base__=base_cls,
        __module__=model_module,
        __validators__=validators,
        **model_fields,
    )

    # Methods must be applied with `setattr` (if simply combined with model_fields, they
    # will not be deserialised)
    for method_name, method_funcdef in methods.items():
        setattr(model_cls, method_name, method_funcdef)
    return model_cls


def extract_validators(
    validators: Union[dict[str, ModelVDeco], dict[str, FieldVDeco]],
) -> Union[ModelVs, FieldVs]:
    return {name: (deco.func, deco.info) for name, deco in validators.items()}


def pickler_building_args(
    model: ModelT,
) -> tuple[
    str, ConfigDict, Optional[str], type[ModelT], str, Fields, ModelVs, FieldVs, Methods
]:
    """Prepare the arguments (which are all passed as positional arguments).

    Keep this function colocated with the `build_pydantic_model` function to ensure it
    is kept up to date with that function's expected signature.

    Arguments:
        model: The `pydantic.BaseModel` subclass instance that triggers the
               deserialisation hook and gets pickled by `build_pydantic_model`.
    """
    decorators = model.__pydantic_decorators__
    model_validators = extract_validators(decorators.model_validators)
    field_validators = {}  # extract_validators(decorators.field_validators)
    model_methods = {
        method_name: model_method
        for method_name, model_method in model.__dict__.items()
        # Private attributes (with `PrivateAttr`) are set through `model_fields`
        # but methods can have also be underscore-prefixed...
        if not method_name.startswith("__")
        # If ABC is an issue, consult other examples like Ray's vendored cloudpickle
        # https://github.com/ray-project/ray/blob/master/python/ray/cloudpickle/cloudpickle.py#L743
        if not method_name.startswith("_abc")
        if method_name
        not in (
            # The `model_*` namespace also includes `model_post_init`, which we
            # do potentially want, so we don't exclude it here.
            "model_fields",
            "model_config",
            *model.model_fields,
            *model_validators,
            *field_validators,
        )
    }

    pickled_model = {
        "name": model.__name__,
        "model_config": model.model_config,
        "model_doc": model.__doc__,
        "base_cls": model.__bases__[0],
        "model_module": model.__module__,
        "model_fields": model.model_fields,
        "model_validators": model_validators,
        "field_validators": field_validators,
        "methods": model_methods,
    }
    pickler_args = tuple(pickled_model.values())
    return pickler_args  # type: ignore


@dill.register(type(BaseModel))
def _dill_hook_for_pydantic_models(
    pickler: dill.Pickler, model: Union[ModelT, type[BaseModel]]
) -> None:
    """Custom dill serialiser for Pydantic models."""
    # TODO: confirm that this definitely receives instances of the model even though the
    # following line indicates it can receive the BaseModel type itself
    if model is BaseModel:
        dill_serialization.save_type(pickler, model)
    else:
        pickler_args = pickler_building_args(model=model)
        pickler.save_reduce(build_pydantic_model, pickler_args)
    return


def deserialise_pydantic_model() -> ModelT:
    """Serialise (`dill.dumps`) then deserialise (`dill.loads`) a Pydantic model.

    The `recurse` setting must be set, counterintuitively, to prevent excessive
    recursion (refer to e.g. dill issue
    [#482](https://github.com/uqfoundation/dill/issues/482#issuecomment-1139017499)):

        to limit the amount of recursion that dill is doing to pickle the function, we
        need to turn on a setting called recurse, but that is because the setting
        actually recurses over the global dictionary and finds the smallest subset that
        the function needs to run, which will limit the number of objects that dill
        needs to include in the pickle.
    """
    dill.settings["recurse"] = True
    serialized_cls = dill.dumps(Input)  # This is picked up magically? RED FLAG
    print("===== DESERIALIZING =====")
    model_cls = dill.loads(serialized_cls)
    deserialised_fvs = vars(model_cls)["__pydantic_decorators__"].field_validators
    pprint(deserialised_fvs)
    print("===== INSTANTIATING =====")
    model = model_cls(prompt="a", num_steps=4, epochs=10)
    return model
