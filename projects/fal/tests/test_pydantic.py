"""Verify that the deserialisation approach works as a standalone test."""
from __future__ import annotations
import subprocess
import sys

import dill
from pydantic import BaseModel, Field, field_validator, model_validator

from pydantic_standalone.pydantic_patch import deserialise_pydantic_model


class Input(BaseModel):
    """A simple Pydantic model used to demonstrate deserialisation via dill.

    Attributes:
        prompt: An input prompt for a generative AI model.
        num_steps: The number of steps to run a generative AI model for.
        validation_counter: A field initialised by default as 0 and incremented by
                            validators (for the purpose of testing).
    """

    prompt: str = ...
    num_steps: int = Field(default=2, ge=1, le=10)
    epochs: int = 10
    validation_counter: int = 0

    def steps_x2(self) -> int:
        """A method which is neither a validator nor provided by Pydantic.

        Computes double of the `num_steps` field value.
        """
        return self.num_steps * 2

    @field_validator("epochs")
    @classmethod
    def triple_epochs(cls, v: int) -> int:
        """A field validator that multiplies the validated field value by 10."""
        raise ValueError("FIELD VALIDATOR RAN")
        return v * 3

    @model_validator(mode="after")
    def increment(self) -> None:
        """A model post-validator that increments a `validation_counter` attribute."""
        self.validation_counter += 100


def validate_deserialisation(model: Input) -> None:
    """Assert the model instance was deserialised correctly.

    Arguments:
        model: The model that was produced by dill deserialisation.
    """
    prompt = model.prompt
    steps = model.num_steps
    steps_x2 = model.steps_x2()
    assert prompt == "a", f"Prompt not retrieved: expected 'a' got {prompt!r}"
    assert steps == 4, f"Steps not retrieved: expected 4 got {steps!r}"
    assert steps_x2 == 8, f"Incorrect `steps_x2()`: expected 8 got {steps_x2}"
    assert model.validation_counter == 100, "The `increment` model validator didn't run"
    # Known bug: only model validators are working at present
    # assert model.epochs == 30, "The `validate_epochs` field validator didn't run"
    return


def deserialise_pydantic_model() -> ModelT:
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
    serialized_cls = dill.dumps(Input)
    print("===== DESERIALIZING =====")
    model_cls = dill.loads(serialized_cls)
    deserialised_fvs = vars(model_cls)["__pydantic_decorators__"].field_validators
    # pprint(deserialised_fvs)
    print("===== INSTANTIATING =====")
    model = model_cls(prompt="a", num_steps=4, epochs=10)
    return model


def test_deserialise_pydantic_model():
    """Test deserialisation of a Pydantic model succeeds.

    The deserialisation failure mode reproduction is incompatible with pytest (see
    [#29](https://github.com/fal-ai/fal/issues/29#issuecomment-1902241217) for
    discussion) so we directly invoke the current Python executable on this file.
    """
    subprocess_args = [sys.executable, __file__, "--run-deserialisation-test"]
    proc = subprocess.run(subprocess_args, capture_output=True, text=True)
    model_fields_ok = "model-field-missing-annotation" not in proc.stderr
    assert model_fields_ok, "Deserialisation failed (`model_fields` not deserialised)"
    # The return code should be zero or else the deserialisation failed
    deserialisation_ok = proc.returncode == 0
    assert deserialisation_ok, f"Pydantic model deserialisation failed:\n{proc.stderr}"


def run_deserialisation_test() -> None:
    """Run a Pydantic model deserialisation and then validate its result is OK.

    Load the code from the `mainify`d `fal._pydantic_patch.patch()` function (which was
    initially developed as a module) to access the `deserialise_pydantic_model` it
    defines. The patch is by definition standalone: this is just serialisation by text!

    **Note** you may want to inspect the field validators to debug further:

        ```
        pprint(vars(Input)["__pydantic_decorators__"].field_validators)
        ```
    """
    model = deserialise_pydantic_model()
    validate_deserialisation(model)


if __name__ == "__main__" and "--run-deserialisation-test" in sys.argv:
    # Reached by pytest subprocess from `test_deserialise_pydantic_model`.
    # Equivalent to calling `python test_pydantic.py --run-deserialisation-test`
    run_deserialisation_test()
