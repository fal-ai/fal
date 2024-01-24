# demo_4_pytest_subprocess.py
import subprocess
import sys

import dill
from pydantic import BaseModel, Field


class Input(BaseModel):
    """A simple Pydantic model used to demonstrate deserialisation via dill.

    Attributes:
        prompt: An input prompt for a generative AI model.
        num_steps: The number of steps to run a generative AI model for.
    """

    prompt: str
    num_steps: int = Field(default=2, ge=1, le=10)


def deserialise_pydantic_model():
    """Serialise (`dill.dumps`) then deserialise (`dill.loads`) a Pydantic model.

    The `recurse` setting must be set, counterintuitively, to prevent excessive
    recursion (refer to e.g. dill issue
    [#482](https://github.com/uqfoundation/dill/issues/482#issuecomment-1139017499)):

        to limit the amount of recursion that dill is doing to pickle the function, we
        need to turn on a setting called recurse, but that is because the setting
        actually recurses over the global dictionary and finds the smallest subset that
        the function needs to run, which will limit the number of objects that dill
        needs to include in the pickle.
    """
    dill.settings["recurse"] = True

    # Run the function locally
    cls = dill.loads(dill.dumps(Input))
    model = cls(prompt="a")
    return model


def test_deserialise_pydantic_model():
    """Test deserialisation of a Pydantic model succeeds.

    The deserialisation failure mode reproduction is incompatible with pytest (see
    [#29](https://github.com/fal-ai/fal/issues/29#issuecomment-1902241217) for
    discussion) so we directly invoke the current Python executable on this file.
    """
    subprocess_args = [sys.executable, __file__, "--run-deserialisation-test"]
    proc = subprocess.run(subprocess_args, capture_output=True, text=True)
    model_fields_ok = "model-field-missing-annotation" in proc.stderr
    assert model_fields_ok, "Deserialisation failed (`model_fields` not deserialised)"
    # The return code should be zero or else the deserialisation failed
    deserialisation_ok = proc.returncode == 0
    assert deserialisation_ok, f"Pydantic model deserialisation failed:\n{proc.stderr}"


if __name__ == "__main__" and "--run-deserialisation-test" in sys.argv:
    deserialise_pydantic_model()
