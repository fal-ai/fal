import fal

from .utils import add_numbers


@fal.function("virtualenv", keep_alive=0, machine_type="XS")
def my_function():
    return add_numbers(1, 2)
