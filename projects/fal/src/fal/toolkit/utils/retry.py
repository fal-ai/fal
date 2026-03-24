import asyncio
import functools
import inspect
import random
import time
import traceback
from typing import Any, Callable, Literal, Optional

BackoffType = Literal["exponential", "fixed"]


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_type: BackoffType = "exponential",
    jitter: bool = False,
    should_retry: Optional[Callable[[Exception], bool]] = None,
) -> Callable:
    def _compute_delay(retries: int) -> float:
        if backoff_type == "exponential":
            delay = min(base_delay * (2 ** (retries - 1)), max_delay)
        else:  # fixed
            delay = min(base_delay, max_delay)

        if jitter:
            delay *= random.uniform(0.5, 1.5)

        return delay

    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                retries = 0
                while retries < max_retries:
                    try:
                        return await func(*args, **kwargs)
                    except asyncio.CancelledError:
                        # Never retry task cancellation.
                        raise
                    except Exception as e:
                        if should_retry is not None and not should_retry(e):
                            raise

                        retries += 1
                        print(f"Retrying {retries} of {max_retries}...")
                        if retries == max_retries:
                            print(f"Max retries reached. Raising exception: {e}")
                            traceback.print_exc()
                            raise

                        await asyncio.sleep(_compute_delay(retries))

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if should_retry is not None and not should_retry(e):
                        raise

                    retries += 1
                    print(f"Retrying {retries} of {max_retries}...")
                    if retries == max_retries:
                        print(f"Max retries reached. Raising exception: {e}")
                        traceback.print_exc()
                        raise

                    time.sleep(_compute_delay(retries))

        return wrapper

    return decorator
