import base64
import datetime
import json
import os
import pickle
import threading
import warnings
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

if TYPE_CHECKING:
    import torch.multiprocessing as mp


def has_type_name(maybe_type: Any, type_name: str) -> bool:
    """
    Checks if the given object has a type name that matches the provided type name.
    This is used to avoid importing torch or other libraries unnecessarily.
    :param maybe_type: The object to check.
    :param type_name: The type name to match against.
    :return: True if the object's type name matches, False otherwise.
    """
    if not isinstance(maybe_type, type):
        maybe_type = type(maybe_type)

    mro_types = [t.__name__ for t in maybe_type.mro()]
    return type_name in mro_types


def is_torch_tensor(obj: Any) -> bool:
    """
    Checks if the given object is a PyTorch tensor without importing torch.
    """
    return has_type_name(obj, "Tensor")


def is_numpy_array(obj: Any) -> bool:
    """
    Checks if the given object is a NumPy array without importing numpy.
    """
    return has_type_name(obj, "ndarray")


def is_pil_image(obj: Any) -> bool:
    """
    Checks if the given object is a PIL Image without importing PIL.
    """
    return has_type_name(obj, "Image")


def format_for_serialization(
    response: Any,
    image_format: str = "jpeg",
    is_final: bool = False,
    as_data_urls: bool = False,
) -> Any:
    """
    Formats the response for serialization.
    Most importantly, it encodes images to base64 and returns the image format and size.
    :param response: The response to format.
    :param is_final: Whether this is the final response.
    :return: The formatted response.
    """
    if is_torch_tensor(response):
        import torch

        with BytesIO() as buffer:
            torch.save(response.detach().cpu(), buffer)
            tensor_bytes = buffer.getvalue()

        if as_data_urls:
            base64_tensor = base64.b64encode(tensor_bytes).decode("utf-8")
            data = None
            url = f"data:application/tensor;base64,{base64_tensor}"
        else:
            data = tensor_bytes
            url = None

        return {
            "content_type": "application/tensor",
            "shape": response.shape,
            "dtype": str(response.dtype),
            "data": data,
            "url": url,
        }
    if is_numpy_array(response):
        import numpy as np

        with BytesIO() as buffer:
            np.save(buffer, response)
            array_bytes = buffer.getvalue()

        if as_data_urls:
            base64_array = base64.b64encode(array_bytes).decode("utf-8")
            data = None
            url = f"data:application/ndarray;base64,{base64_array}"
        else:
            data = array_bytes
            url = None

        return {
            "content_type": "application/ndarray",
            "shape": response.shape,
            "dtype": str(response.dtype),
            "data": data,
            "url": url,
        }
    elif is_pil_image(response):
        width, height = response.size

        with BytesIO() as buffer:
            if is_final:
                if image_format == "jpeg":
                    response.save(buffer, format="jpeg", quality=95)
                else:
                    response.save(buffer, format=image_format)
            else:
                image_format = "jpeg"
                response.save(buffer, format="jpeg", quality=60)

            image_bytes = buffer.getvalue()

        if as_data_urls:
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            url = f"data:image/{image_format};base64,{base64_image}"
            data = None
        else:
            url = None
            data = image_bytes

        return {
            "content_type": f"image/{image_format}",
            "width": width,
            "height": height,
            "data": data,
            "url": url,
        }
    elif isinstance(response, list):
        return [
            format_for_serialization(
                item,
                image_format=image_format,
                is_final=is_final,
                as_data_urls=as_data_urls,
            )
            for item in response
        ]
    elif isinstance(response, dict):
        return {
            key: format_for_serialization(
                value,
                image_format=image_format,
                is_final=is_final,
                as_data_urls=as_data_urls,
            )
            for key, value in response.items()
        }

    return response


def format_deserialized_data(data: Any) -> Any:
    """
    Formats the deserialized data for further processing.
    :param data: The data to format.
    :return: The formatted data.
    """
    if isinstance(data, dict):
        if data.get("content_type", "").startswith("image/"):
            from PIL import Image

            # Deserialize image data
            if data.get("url"):
                # If the data is a data URL, decode it
                base64_image = data["url"].split(",")[1]
                image_bytes = base64.b64decode(base64_image)
            elif data.get("data"):
                image_bytes = data["data"]
            else:
                raise ValueError("Image data must contain either 'url' or 'data'.")

            fp = BytesIO(image_bytes)  # Don't close the BytesIO object
            return Image.open(fp)
        elif data.get("content_type", "") == "application/tensor":
            import torch

            # Deserialize tensor data
            if data.get("url"):
                # If the data is a data URL, decode it
                base64_tensor = data["url"].split(",")[1]
                tensor_bytes = base64.b64decode(base64_tensor)
            elif data.get("data"):
                tensor_bytes = data["data"]
            else:
                raise ValueError("Tensor data must contain either 'url' or 'data'.")

            with BytesIO(tensor_bytes) as buffer:
                return torch.load(buffer)

        elif data.get("content_type", "") == "application/ndarray":
            import numpy as np

            # Deserialize numpy array data
            if data.get("url"):
                # If the data is a data URL, decode it
                base64_array = data["url"].split(",")[1]
                array_bytes = base64.b64decode(base64_array)
            elif data.get("data"):
                array_bytes = data["data"]
            else:
                raise ValueError(
                    "Numpy array data must contain either 'url' or 'data'."
                )

            with BytesIO(array_bytes) as buffer:
                return np.load(buffer, allow_pickle=True)

        return {key: format_deserialized_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [format_deserialized_data(item) for item in data]
    return data


def distributed_serialize(
    obj: Any, is_final: bool = False, image_format: str = "jpeg"
) -> bytes:
    """
    Serializes an object to a JSON string.
    :param obj: The object to serialize.
    :return: The serialized JSON string.
    """
    data = format_for_serialization(obj, is_final=is_final, image_format=image_format)
    return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)


def encode_text_event(
    obj: Any, is_final: bool = False, image_format: str = "jpeg"
) -> bytes:
    """
    Encodes a text response as a JSON string.
    :param response: The text response to encode.
    :param is_final: Whether this is the final response.
    :return: The encoded JSON string.
    """
    formatted = format_for_serialization(
        obj, image_format=image_format, is_final=is_final, as_data_urls=True
    )
    return f"data: {json.dumps(formatted)}\n\n".encode()


def distributed_deserialize(serialized: Union[bytes, str]) -> Any:
    """
    Deserializes a JSON string to an object.
    :param serialized: The serialized JSON string.
    :return: The deserialized object.
    """
    if isinstance(serialized, str):
        data = json.loads(serialized)
    else:
        data = pickle.loads(serialized)
    return format_deserialized_data(data)


def wrap_distributed_worker(
    rank: int,
    func: Callable,
    world_size: int,
    master_addr: str,
    master_port: int,
    timeout: int,
    cwd: Optional[Union[str, Path]],
    args: Tuple[Any],
    kwargs: Dict[str, Any],
) -> None:
    """
    Worker function for distributed training or inference.

    This function is called by each worker process spawned by
    `torch.multiprocessing.spawn`.

    :param func: The function to run in each worker process.
    :param world_size: The total number of processes.
    :param rank: The rank of the current process.
    :param master_addr: The address of the master node.
    :param master_port: The port on which the master node will listen.
    """
    import torch
    import torch.distributed as dist

    if cwd:
        os.chdir(str(cwd))

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    print(f"[debug] Worker {rank} started with PID {os.getpid()}.")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=timeout),
        device_id=torch.device(f"cuda:{rank}"),
    )

    try:
        func(*args, **kwargs)
    finally:
        dist.destroy_process_group()


def launch_distributed_processes(
    func: Callable,
    world_size: int = 1,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    timeout: int = 1800,
    cwd: Optional[Union[str, Path]] = None,
    *args: Any,
    **kwargs: Any,
) -> "mp.ProcessContext":
    """
    Launches a distributed process group using torch.multiprocessing.spawn.
    This function is designed to be called from the main process and will
    spawn multiple worker processes for distributed training or inference.
    :param func: The function to run in each worker process.
    :param world_size: The total number of processes to spawn.
    :param master_addr: The address of the master node.
    :param master_port: The port on which the master node will listen.
    :return: The process context for the spawned processes.
    """

    import torch.distributed as dist

    if dist.is_initialized():
        raise RuntimeError(
            "Distributed process group is already initialized. "
            "Please ensure that you are not trying to initialize it multiple times."
        )

    import torch.multiprocessing as mp

    try:
        import cloudpickle  # type: ignore[import-untyped]

        mp.reducer.ForkingPickler = cloudpickle.Pickler  # type: ignore[misc]
    except ImportError:
        warnings.warn(
            "Could not import cloudpickle, using default pickler. "
            "If you encounter pickling errors, please install cloudpickle.",
        )

    print(f"[debug] Launching distributed processes with world size {world_size}.")
    return mp.spawn(  # type: ignore[no-untyped-call]
        wrap_distributed_worker,
        args=(
            func,
            world_size,
            master_addr,
            master_port,
            timeout,
            cwd,
            args,
            kwargs,
        ),
        nprocs=world_size,
        join=False,
    )


class KeepAliveTimer:
    """
    Call a function after a certain amount of time to keep the worker alive.
    """

    timer: Optional[threading.Timer]

    def __init__(
        self,
        func: Callable,
        timeout: Union[int, float],
        start: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.func = func
        self.timeout = timeout
        self.timer = None
        self.args = args
        self.kwargs = kwargs
        self.timer = None
        if start:
            self.start()

    def start(self) -> None:
        """
        Start the timer.
        """
        self.timer = threading.Timer(self.timeout, self.func, self.args, self.kwargs)
        self.timer.start()

    def cancel(self) -> None:
        """
        Cancel the timer.
        """
        if self.timer:
            self.timer.cancel()
            self.timer = None

    def reset(self) -> None:
        """
        Reset the timer.
        """
        self.cancel()
        self.start()
