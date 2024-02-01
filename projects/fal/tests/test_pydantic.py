# demo_4_pytest_subprocess.py
import subprocess
import sys
from pprint import pprint
from pydantic import BaseModel, Field, field_validator, model_validator

from pydantic_patch import deserialise_pydantic_model


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
    assert model.epochs == 30, "The `validate_epochs` field validator didn't run"
    assert model.validation_counter == 100, "The `increment` model validator didn't run"
    return


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


if __name__ == "__main__" and "--run-deserialisation-test" in sys.argv:
    pprint(vars(Input)["__pydantic_decorators__"].field_validators)
    model = deserialise_pydantic_model()
    validate_deserialisation(model)
