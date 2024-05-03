from __future__ import annotations

import os
import traceback
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def optimize(
    module: torch.nn.Module, *, optimization_config: dict[str, Any] | None = None
) -> torch.nn.Module:
    """Optimize the given torch module with dynamic compilation and
    quantization techniques. Only applicable under fal's cloud environment.

    Warning: This function is experimental and may not work as expected.
    """
    import runpy

    import torch.nn as nn

    if not isinstance(module, nn.Module):
        raise TypeError(f"Expected a torch.nn.Module, got {type(module)}.")

    optimizer_path = os.environ.get("FAL_SPATIAL_OPTIMIZER", None)
    if optimizer_path is None:
        print(
            "[WARNING] FAL_SPATIAL_OPTIMIZER is not set, falling back"
            "to default torch execution"
        )
        return module

    try:
        spatial_optimizer = runpy.run_path(optimizer_path, run_name="spatial_optimizer")

        return spatial_optimizer["optimize"](
            module,
            optimization_config=optimization_config,
        )
    except Exception:
        print(
            "[WARNING] Failed to optimize module, falling back "
            "to default torch execution."
        )
        traceback.print_exc()
        return module
