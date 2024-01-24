# demo_4_pytest_subprocess.py
import subprocess
import sys

import dill
import dill._dill as dill_serialization
from pydantic import BaseModel, Field, model_validator


def build_pydantic_model(
    name, base_cls, model_config, model_fields, validators, class_fields
):
    """Recreate the Pydantic model from the deserialised validator info."""
    import pydantic

    model_cls = pydantic.create_model(
        name,
        __base__=base_cls,
        __validators__={
            validator_name: pydantic.model_validator(mode=validator_info.mode)(
                validator_func
            )
            for validator_name, (validator_func, validator_info) in validators.items()
        },
        **model_fields,
        **class_fields,
    )
    return model_cls


@dill.register(type(BaseModel))
def _dill_hook_for_pydantic_models(pickler: dill.Pickler, pydantic_model) -> None:
    if pydantic_model is BaseModel:
        dill_serialization.save_type(pickler, pydantic_model)
        return

    validators = {}
    decorators = pydantic_model.__pydantic_decorators__
    for validator_name, decorator in decorators.model_validators.items():
        validators[validator_name] = (decorator.func, decorator.info)

    class_fields = {
        "__annotations__": pydantic_model.__annotations__,
    }
    for class_field_name, class_field_value in pydantic_model.__dict__.items():
        if class_field_name.startswith("_"):
            continue
        elif class_field_name in ("model_fields", "model_config"):
            continue
        elif class_field_name in pydantic_model.model_fields:
            continue
        elif class_field_name in validators:
            continue

        class_fields[class_field_name] = class_field_value

    pickler.save_reduce(
        build_pydantic_model,
        (
            pydantic_model.__name__,
            pydantic_model.__bases__[0],
            pydantic_model.model_config,
            pydantic_model.model_fields,
            validators,
            class_fields,
        ),
    )


class Input(BaseModel):
    """A simple Pydantic model used to demonstrate deserialisation via dill.

    Attributes:
        prompt: An input prompt for a generative AI model.
        num_steps: The number of steps to run a generative AI model for.
        validation_counter: A field initialised by default as 0 and incremented by
                            validators (for the purpose of testing).
    """

    prompt: str = Field()
    num_steps: int = Field(default=2, ge=1, le=10)
    validation_counter: int = 0

    def steps_x2(self) -> int:
        """A method which is neither a validator nor provided by Pydantic.

        Computes double of the `num_steps` field value."""
        return self.num_steps * 2

    @model_validator(mode="after")
    def validate_num_steps(self) -> None:
        """A model post-validator."""
        self.validation_counter += 100


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
    serialized_cls = dill.dumps(Input)
    print("====== DESERIALIZING =====")
    model_cls = dill.loads(serialized_cls)
    print("======== RUNNING =====")
    model = model_cls(prompt="a")
    return model


def validate_deserialisation(model: Input) -> None:
    prompt = model.prompt
    steps = model.num_steps
    steps_x2 = model.steps_x2()
    assert prompt == "a", f"Prompt not retrieved: expected 'a' got {prompt!r}"
    assert steps == 2, f"Steps not retrieved: expected 2 got {steps!r}"
    assert steps_x2 == 4, f"Incorrect `steps_x2()`: expected 4 got {steps_x2}"
    assert model.validation_counter == 100
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
    model = deserialise_pydantic_model()
    validate_deserialisation(model)
