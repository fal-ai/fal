# `fal dev hot` example: FLUX.2 [klein] 4B

`load.py` loads the pipeline onto the GPU once; `infer.py` and `post.py` reload on every save, with the weights kept loaded. `requirements.txt` is installed into the folder's `.venv` on the box, on top of the image's packages.

```bash
fal dev up
fal dev hot start ./examples/dev_hot_klein
fal dev hot run --prompt "a red fox in snow" -p steps=4 -o fox.png
fal dev hot sync --watch --run      # edit infer.py, save, see the new image
fal dev down
```

Saving `load.py` or `requirements.txt` restarts the server and reloads the weights.
