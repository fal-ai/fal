"""Patch for serialising Pydantic v2 models."""
from __future__ import annotations

from typing_extensions import TypeAlias  # 3.9+ can import from typing

from fal.toolkit import mainify

__all__ = ["patch"]


@mainify
def patch():
    from typing import Callable, TypeVar

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
        "pickler_building_args",
        "_dill_hook_for_pydantic_models",
        "deserialise_pydantic_model",
        "validate_deserialisation",
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
    ModelVDeco: TypeAlias = Decorator[ModelValidatorDecoratorInfo]
    """Alias for the decorator storing info about a model validator."""
    FieldVDeco: TypeAlias = Decorator[FieldValidatorDecoratorInfo]
    """Alias for the decorator storing info about a field validator."""

    def build_pydantic_model(
        name: str,
        model_config: ConfigDict,
        model_doc: str | None,
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
            name: getattr(field, "annotation", None)
            for name, field in model_fields.items()
        }
        model_fields.update(
            {name: func for name, (func, _) in field_validators.items()}
        )

        model_cls = pydantic.create_model(
            name,
            # __config__=model_config, # UNUSED
            __doc__=model_doc,
            __base__=base_cls,
            __module__=model_module,
            __validators__=validators,
            **model_fields,
        )

        # Methods must be applied with `setattr` (if simply combined with model_fields,
        # they will not be deserialised)
        for method_name, method_funcdef in methods.items():
            setattr(model_cls, method_name, method_funcdef)
        return model_cls

    def extract_validators(
        validators: dict[str, ModelVDeco] | dict[str, FieldVDeco],
    ) -> ModelVs | FieldVs:
        return {
            name: (deco.func.__func__, deco.info) for name, deco in validators.items()
        }

    def pickler_building_args(
        model: ModelT,
    ) -> tuple[
        str,
        ConfigDict,
        str | None,
        type[ModelT],
        str,
        Fields,
        ModelVs,
        FieldVs,
        Methods,
    ]:
        """Prepare the arguments (which are all passed as positional arguments).

        Keep this function colocated with the `build_pydantic_model` function to ensure it
        is kept up to date with that function's expected signature.

        Arguments:
            model: The `pydantic.BaseModel` subclass instance that triggers the
                deserialisation hook and gets pickled by `build_pydantic_model`.
        """
        from pprint import pprint

        decorators = model.__pydantic_decorators__
        model_validators = extract_validators(decorators.model_validators)
        field_validators = {}  # extract_validators(decorators.field_validators)
        model_methods = {}

        # model_base_cls = model.__mro__[0]
        # base_model_cls = model.__mro__[1]

        # print(f"CLASS KEYS: ({model_base_cls})")
        # model_class_dict_keys = list(vars(model_base_cls))
        # pprint(model_class_dict_keys)

        # print(f"BASE CLASS KEYS: ({base_model_cls})")
        # base_model_class_dict_keys = list(vars(base_model_cls))
        # pprint(base_model_class_dict_keys)

        # print(f"INSTANCE KEYS: ({model})")
        # model_instance_dict_keys = list(vars(model))
        # pprint(model_instance_dict_keys)

        # base_only = set(base_model_class_dict_keys).difference(model_class_dict_keys)
        # model_only = set(model_class_dict_keys).difference(base_model_class_dict_keys)
        # both_bm = set(base_model_class_dict_keys).union(model_class_dict_keys)
        # both_bm_no_pyd = [k for k in both_bm if not k.startswith("__pydantic")]

        # Unsure if we actually need to traverse the MRO... Maybe for CompressedFile?
        # for model_base_class in model.__mro__:
        model_qualname = model.__qualname__
        for method_name, model_method in model.__dict__.items():
            if method_name == "__weakref__":
                # This is added specifically by Pydantic for cloudpickle, do not touch
                continue
            if not hasattr(model_method, "__qualname__"):
                print(f"SKIPPING {method_name}")
                continue
            expected_method_qualname = f"{model_qualname}.{method_name}"
            own_method = model_method.__qualname__ == expected_method_qualname
            if not own_method:
                print(f"NOT OWNED: {method_name}")
                continue
            print(f"GOT {method_name}")
            # if method_name.startswith("__pydantic_"):
            #     # Leave Pydantic's private namespace alone!
            #     continue
            # elif method_name.startswith("_abc"):
            #     # If ABC is an issue, consult other examples like Ray's vendored cloudpickle
            #     # https://github.com/ray-project/ray/blob/master/python/ray/cloudpickle/cloudpickle.py#L743
            #     continue
            # elif method_name in ("model_fields", "model_config"):
            #     # The `model_*` namespace also includes `model_post_init`, which we do
            #     # potentially want, so we don't exclude it here.
            #     continue
            if method_name in model.model_fields:
                continue
            elif method_name in model_validators:
                continue
            elif method_name in field_validators:
                continue
            model_methods.update({method_name: model_method})

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
        pickler: dill.Pickler, model: ModelT | type[BaseModel]
    ) -> None:
        """Custom dill serialiser for Pydantic models."""
        # TODO: confirm that this definitely receives instances of the model even though the
        # following line indicates it can receive the BaseModel type itself
        if model is BaseModel:
            dill_serialization.save_type(pickler, model)
        else:
            # model is a `pydantic._internal._model_construction.ModelMetaclass`
            pickler_args = pickler_building_args(model=model)
            pickler.save_reduce(build_pydantic_model, pickler_args)
        return
