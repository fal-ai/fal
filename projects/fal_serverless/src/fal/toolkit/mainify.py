from __future__ import annotations


# HACK: make classes dillable https://github.com/uqfoundation/dill/issues/424
# Only works for classes for now, must be outer-most decorator in most cases
def mainify(obj):
    if hasattr(obj, "__module__") and obj.__module__.startswith("fal"):
        obj.__module__ = "__main__"

        for inner in obj.__dict__.values():
            mainify(inner)

    return obj
