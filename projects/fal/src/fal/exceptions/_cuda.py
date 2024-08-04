from __future__ import annotations

from dataclasses import dataclass

from ._base import AppException

# PyTorch error message for out of memory
_CUDA_OOM_MESSAGE = "CUDA error: out of memory"

# Special status code for CUDA out of memory errors
_CUDA_OOM_STATUS_CODE = 503


@dataclass
class CUDAOutOfMemoryException(AppException):
    """Exception raised when a CUDA operation runs out of memory."""

    message: str = _CUDA_OOM_MESSAGE
    status_code: int = _CUDA_OOM_STATUS_CODE


# based on https://github.com/Lightning-AI/pytorch-lightning/blob/37e04d075a5532c69b8ac7457795b4345cca30cc/src/lightning/pytorch/utilities/memory.py#L49
def _is_cuda_oom_exception(exception: BaseException) -> bool:
    return _is_cuda_out_of_memory(exception) or _is_cudnn_snafu(exception)


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def _is_cuda_out_of_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def _is_cudnn_snafu(exception: BaseException) -> bool:
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )
