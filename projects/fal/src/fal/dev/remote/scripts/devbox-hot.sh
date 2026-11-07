#!/bin/bash
# Usage: devbox-hot <project folder>
# Serves load.py and infer.py of the project on 127.0.0.1:$PORT. The server exits
# with code $RESTART_CODE when load.py or the requirements change; it is then
# started again, which reloads the weights. Requirements are reinstalled first,
# constrained to the base image's torch stack, into a .venv that is rebuilt when
# requirements.txt changes (and removed when it is emptied or deleted).
# Started by `fal dev hot start` in tmux. box_app.py fills in the capitalized
# names when it installs this script.
cd "$1" || exit 1
reqs_stamp=.venv/.devbox-requirements
constraints=${DEVBOX_HOT_CONSTRAINTS:-/tmp/devbox-hot-constraints.txt}
# Pins the base image's torch stack (torch, torchvision, torchaudio, triton and the
# nvidia-*-cuNN libraries) so a requirement cannot pull a newer torch into the .venv, which
# breaks the base's torchvision ("operator torchvision::nms does not exist").
write_constraints() {
    python3 - "$constraints" <<'PY'
import importlib.metadata as md
import re
import sys

STACK = {"torch", "torchvision", "torchaudio", "triton"}
pins = set()
for dist in md.distributions():
    name = dist.metadata["Name"] or ""
    key = re.sub(r"[-_.]+", "-", name).lower()
    cuda_lib = key.startswith("nvidia-") and key.endswith(("-cu11", "-cu12", "-cu13"))
    if key in STACK or cuda_lib:
        pins.add(f"{key}=={dist.version.split('+')[0]}")
with open(sys.argv[1], "w") as out:
    out.write("\n".join(sorted(pins)) + "\n")
PY
}
while true; do
    # The requirements that count: none when the file is missing, empty or comments.
    want=none
    if [ -f requirements.txt ] && grep -qvE '^[[:space:]]*(#|$)' requirements.txt; then
        want=$(grep -vE '^[[:space:]]*(#|$)' requirements.txt | sha256sum | cut -d' ' -f1)
    fi
    # Rebuild a .venv this script made when the requirements changed since. A .venv
    # without a stamp is the user's own (or predates the stamp) and is left alone.
    if [ -f "$reqs_stamp" ] && [ "$(cat "$reqs_stamp")" != "$want" ]; then
        echo "[hot] requirements.txt changed; rebuilding .venv"
        rm -rf .venv
    fi
    if [ "$want" != none ]; then
        if [ ! -x .venv/bin/python ]; then
            uv venv -q --system-site-packages .venv
            # Stamped at birth, so a failed first install is still ours to rebuild.
            echo pending > "$reqs_stamp"
        fi
        before=$(cksum < requirements.txt)
        if ! write_constraints || ! uv pip install -q --python .venv/bin/python -c "$constraints" -r requirements.txt; then
            echo "[hot] installing requirements.txt failed; fix it and save"
            # Never leave a half-built .venv for the server to run in.
            [ -f "$reqs_stamp" ] && rm -rf .venv
            while [ "$(cksum < requirements.txt 2>/dev/null)" = "$before" ]; do sleep 1; done
            continue
        fi
        [ -f "$reqs_stamp" ] && echo "$want" > "$reqs_stamp"
    fi
    py=python3
    [ -x .venv/bin/python ] && py=.venv/bin/python
    "$py" -u $SERVER . --port $PORT --restart-code $RESTART_CODE
    code=$?
    if [ "$code" -ne $RESTART_CODE ]; then
        echo "[hot] server exited with code $code"
        exit "$code"
    fi
    echo "[hot] restarting"
done
