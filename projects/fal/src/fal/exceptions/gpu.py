from __future__ import annotations

import contextlib
from dataclasses import dataclass

from ._base import AppException

# PyTorch error message for out of memory
_CUDA_OOM_MESSAGE = "CUDA error: out of memory"

# Special status code for GPU errors
_GPU_ERROR_STATUS_CODE = 503

# GPU error markers
_GPU_ERROR_MARKERS = (
    "cuda",
    "cudnn",
    "nvml",
    "nccl",
    "cublas",
    "cufft",
    "cusolver",
    "cusparse",
    "triton",
)


@dataclass
class GPUException(AppException):
    """Base exception for GPU-related errors."""

    message: str = "GPU error"
    status_code: int = _GPU_ERROR_STATUS_CODE


@dataclass
class GPUOutOfMemoryException(GPUException):
    """Exception raised when a GPU operation runs out of memory."""


@dataclass
class CUDAOutOfMemoryException(GPUOutOfMemoryException):
    """Exception raised when a CUDA operation runs out of memory."""

    message: str = _CUDA_OOM_MESSAGE


# based on https://github.com/Lightning-AI/pytorch-lightning/blob/37e04d075a5532c69b8ac7457795b4345cca30cc/src/lightning/pytorch/utilities/memory.py#L49
def _is_cuda_oom_exception(exception: BaseException) -> bool:
    return _is_cuda_out_of_memory(exception) or _is_cudnn_snafu(exception)


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def _is_cuda_out_of_memory(exception: BaseException) -> bool:
    if not isinstance(exception, RuntimeError) or len(exception.args) != 1:
        return False

    msg = exception.args[0]

    if "CUDA" in msg and "out of memory" in msg:
        return True

    # https://github.com/pytorch/pytorch/issues/112377
    if "NVML_SUCCESS == r INTERNAL ASSERT FAILED" in msg:
        return True

    if "CUDNN_STATUS_INTERNAL_ERROR" in msg:
        return True

    return False


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def _is_cudnn_snafu(exception: BaseException) -> bool:
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


def _is_generic_gpu_error(exception: BaseException) -> bool:
    """Broad marker-based check for GPU-related errors as last resort."""
    text = str(exception).lower()
    return any(marker in text for marker in _GPU_ERROR_MARKERS)


class catch_gpu_exceptions(contextlib.ContextDecorator):
    """Catch GPU/CUDA exceptions and convert them to HTTP 503 responses.

    Works as both a context manager and a decorator. Any caught GPU
    exception (CUDA OOM, cuDNN errors, NVML failures, etc.) is
    re-raised as a GPU exception with HTTP status 503.

    Raises:
        CUDAOutOfMemoryException: When a CUDA OOM error is caught.
        GPUException: When a generic GPU-related error is caught.

    Example:
        from fal.exceptions import catch_gpu_exceptions

        with catch_gpu_exceptions():
            run_inference()

        @catch_gpu_exceptions()
        def run_inference():
            ...

    Note:
        The 503 status code signals the platform to restart the runner.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            return False

        if isinstance(exc_val, GPUException):
            return False

        if _is_cuda_oom_exception(exc_val):
            raise CUDAOutOfMemoryException() from exc_val

        if _is_generic_gpu_error(exc_val):
            raise GPUException() from exc_val

        return False
