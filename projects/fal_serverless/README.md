# fal-serverless

Library to run, serve or schedule your Python functions in the cloud with any machine type you may need.

Check out to the [docs](https://docs.fal.ai/fal-serverless/quickstart) for more details.

## Generate OpenAPI client for the REST API

Initial client was generated using

```sh
cd projects/fal_serverless
# Notice that you can point to any environment
openapi-python-client generate --url https://rest.shark.fal.ai/openapi.json
```

### Update client

```sh
cd projects/fal_serverless
# Notice that you can point to any environment
openapi-python-client update --url https://rest.shark.fal.ai/openapi.json
```
