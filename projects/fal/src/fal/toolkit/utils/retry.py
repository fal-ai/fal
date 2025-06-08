import functools
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
    def decorator(func: Callable) -> Callable:
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

                        raise e

                    if backoff_type == "exponential":
                        delay = min(base_delay * (2 ** (retries - 1)), max_delay)
                    else:  # fixed
                        delay = min(base_delay, max_delay)

                    if jitter:
                        delay *= random.uniform(0.5, 1.5)

                    time.sleep(delay)

        return wrapper

    return decorator
