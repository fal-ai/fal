# gRPC definitions

For regenerating definitions:

```
$ cd projects/isolate_proto
$ python ../../tools/regen_grpc.py src/isolate_proto/controller.proto <isolate version>
$ pre-commit run --all-files
```

The `<isolate version>` argument needs to be a [tag from the isolate Github project](https://github.com/fal-ai/isolate/tags) minus the leading `v`.
