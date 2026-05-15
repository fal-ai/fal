from __future__ import annotations

from typing import ForwardRef

import cloudpickle

from fal._serialization import patch_pickle


def test_forward_ref_serialization_drops_patch_version_cache():
    patch_pickle()

    ref = ForwardRef("App")
    if hasattr(ref, "__resolved_str_cache__"):
        ref.__resolved_str_cache__ = "App"

    payload = cloudpickle.dumps(ref)

    assert b"__resolved_str_cache__" not in payload
    restored = cloudpickle.loads(payload)
    assert restored.__forward_arg__ == ref.__forward_arg__
