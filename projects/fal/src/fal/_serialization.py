from __future__ import annotations

import pickle
from typing import Any, Callable

import cloudpickle


def _register_pickle_by_value(name) -> None:
    # cloudpickle.register_pickle_by_value wants an imported module object,
    # but there is really no reason to go through that complication, as
    # it might be prone to errors.
    cloudpickle.cloudpickle._PICKLE_BY_VALUE_MODULES.add(name)


def include_package_from_path(raw_path: str) -> None:
    from pathlib import Path

    path = Path(raw_path).resolve()
    parent = path
    while (parent.parent / "__init__.py").exists():
        parent = parent.parent

    if parent != path:
        _register_pickle_by_value(parent.name)


def include_modules_from(obj: Any) -> None:
    module_name = getattr(obj, "__module__", None)
    if not module_name:
        return

    if "." in module_name:
        # Just include the whole package
        package_name, *_ = module_name.partition(".")
        _register_pickle_by_value(package_name)
        return

    if module_name == "__main__":
        # When the module is __main__, we need to recursively go up the
        # tree to locate the actual package name.
        import __main__

        include_package_from_path(__main__.__file__)
        return

    _register_pickle_by_value(module_name)


def _register(cls: Any, func: Callable) -> None:
    cloudpickle.Pickler.dispatch[cls] = func


def _patch_pydantic_field_serialization() -> None:
    # Cythonized pydantic fields can't be serialized automatically, so we are
    # have a special case handling for them that unpacks it to a dictionary
    # and then reloads it on the other side.
    # https://github.com/ray-project/ray/blob/842bbcf4236e41f58d25058b0482cd05bfe9e4da/python/ray/_private/pydantic_compat.py#L80
    try:
        from pydantic.fields import ModelField, ModelPrivateAttr
    except ImportError:
        return

    def create_model_field(kwargs: dict) -> ModelField:
        return ModelField(**kwargs)

    def pickle_model_field(field: ModelField) -> tuple[Callable, tuple]:
        kwargs = {
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

        return create_model_field, (kwargs,)

    def create_private_attr(kwargs: dict) -> ModelPrivateAttr:
        return ModelPrivateAttr(**kwargs)

    def pickle_private_attr(field: ModelPrivateAttr) -> tuple[Callable, tuple]:
        kwargs = {
            "default": field.default,
            "default_factory": field.default_factory,
        }

        return create_private_attr, (kwargs,)

    _register(ModelField, pickle_model_field)
    _register(ModelPrivateAttr, pickle_private_attr)


def _patch_pydantic_model_serialization() -> None:
    # If user has created new pydantic models in his namespace, we will try to pickle
    # those by value, which means recreating class skeleton, which will stumble upon
    # __pydantic_parent_namespace__ in its __dict__ and it may contain modules that
    # happened to be imported in the namespace but are not actually used, resulting
    # in pickling errors. Unfortunately this also means that `model_rebuid()` might
    # not work.
    try:
        import pydantic
    except ImportError:
        return

    # https://github.com/pydantic/pydantic/pull/2573
    if not hasattr(pydantic, "__version__") or pydantic.__version__.startswith("1."):
        return

    backup = "_original_extract_class_dict"
    if getattr(cloudpickle.cloudpickle, backup, None):
        return

    original = cloudpickle.cloudpickle._extract_class_dict

    def patched(cls):
        attr_name = "__pydantic_parent_namespace__"
        if issubclass(cls, pydantic.BaseModel) and getattr(cls, attr_name, None):
            setattr(cls, attr_name, None)

        return original(cls)

    cloudpickle.cloudpickle._extract_class_dict = patched
    setattr(cloudpickle.cloudpickle, backup, original)


def _patch_lru_cache() -> None:
    # https://github.com/cloudpipe/cloudpickle/issues/178
    # https://github.com/uqfoundation/dill/blob/70f569b0dd268d2b1e85c0f300951b11f53c5d53/dill/_dill.py#L1429

    from functools import _lru_cache_wrapper as LRUCacheType
    from functools import lru_cache

    def create_lru_cache(func: Callable, kwargs: dict) -> LRUCacheType:
        return lru_cache(**kwargs)(func)

    def pickle_lru_cache(obj: LRUCacheType) -> tuple[Callable, tuple]:
        if hasattr(obj, "cache_parameters"):
            params = obj.cache_parameters()
            kwargs = {
                "maxsize": params["maxsize"],
                "typed": params["typed"],
            }
        else:
            kwargs = {"maxsize": obj.cache_info().maxsize}

        return create_lru_cache, (obj.__wrapped__, kwargs)

    _register(LRUCacheType, pickle_lru_cache)


def _patch_lock() -> None:
    # https://github.com/uqfoundation/dill/blob/70f569b0dd268d2b1e85c0f300951b11f53c5d53/dill/_dill.py#L1310
    from _thread import LockType
    from threading import Lock

    def create_lock(locked: bool) -> Lock:
        lock = Lock()
        if locked and not lock.acquire(False):
            raise pickle.UnpicklingError("Cannot acquire lock")
        return lock

    def pickle_lock(obj: LockType) -> tuple[Callable, tuple]:
        return create_lock, (obj.locked(),)

    _register(LockType, pickle_lock)


def _patch_rlock() -> None:
    # https://github.com/uqfoundation/dill/blob/70f569b0dd268d2b1e85c0f300951b11f53c5d53/dill/_dill.py#L1317
    from _thread import RLock as RLockType  # type: ignore[attr-defined]

    def create_rlock(count: int, owner: int) -> RLockType:
        lock = RLockType()
        if owner is not None:
            lock._acquire_restore((count, owner))  # type: ignore[attr-defined]
        if owner and not lock._is_owned():  # type: ignore[attr-defined]
            raise pickle.UnpicklingError("Cannot acquire lock")
        return lock

    def pickle_rlock(obj: RLockType) -> tuple[Callable, tuple]:
        r = obj.__repr__()
        count = int(r.split("count=")[1].split()[0].rstrip(">"))
        owner = int(r.split("owner=")[1].split()[0])

        return create_rlock, (count, owner)

    _register(RLockType, pickle_rlock)


def _patch_console_thread_locals() -> None:
    try:
        from rich.console import ConsoleThreadLocals
    except ModuleNotFoundError:
        return

    def create_locals(kwargs: dict) -> ConsoleThreadLocals:
        return ConsoleThreadLocals(**kwargs)

    def pickle_locals(obj: ConsoleThreadLocals) -> tuple[Callable, tuple]:
        kwargs = {
            "theme_stack": obj.theme_stack,
            "buffer": obj.buffer,
            "buffer_index": obj.buffer_index,
        }
        return create_locals, (kwargs,)

    _register(ConsoleThreadLocals, pickle_locals)


def _patch_exceptions() -> None:
    # Support chained exceptions
    from tblib.pickling_support import install

    install()


def patch_pickle() -> None:
    _patch_pydantic_field_serialization()
    _patch_pydantic_model_serialization()
    _patch_lru_cache()
    _patch_lock()
    _patch_rlock()
    _patch_console_thread_locals()
    _patch_exceptions()

    _register_pickle_by_value("fal")
