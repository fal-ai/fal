# fal

fal is a serverless Python runtime that lets you run and scale code in the cloud with no infra management.

With fal, you can build pipelines, serve ML models and scale them up to many users. You scale down to 0 when you don't use any resources.

## Quickstart

First, you need to install the `fal` package. You can do so using pip:
```shell
pip install fal
```

Then you need to authenticate:
```shell
fal auth login
```

You can also use fal keys that you can get from [our dashboard](https://fal.ai/dashboard/keys).

Now can use the fal package in your Python scripts as follows:

```py
import fal

@fal.function(
    "virtualenv",
    requirements=["pyjokes"],
)
def tell_joke() -> str:
    import pyjokes

    joke = pyjokes.get_joke()
    return joke

print("Joke from the clouds: ", tell_joke())
```

A new virtual environment will be created by fal in the cloud and the set of requirements that we passed will be installed as soon as this function is called. From that point on, our code will be executed as if it were running locally, and the joke prepared by the pyjokes library will be returned.

## Next steps

If you would like to find out more about the capabilities of fal, check out to the [docs](https://fal.ai/docs). You can learn more about persistent storage, function caches and deploying your functions as API endpoints.
