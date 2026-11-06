#!/bin/bash
# Usage: devbox-hot <project folder>
# Serves load.py and infer.py of the project on 127.0.0.1:$PORT. The server exits
# with code $RESTART_CODE when load.py or the requirements change; it is then
# started again, which reloads the weights. Requirements are reinstalled first.
# Started by `fal dev hot start` in tmux. box_app.py fills in the capitalized
# names when it installs this script.
cd "$1" || exit 1
while true; do
    if [ -f requirements.txt ]; then
        [ -x .venv/bin/python ] || uv venv -q --system-site-packages .venv
        if ! uv pip install -q --python .venv/bin/python -r requirements.txt; then
            echo "[hot] installing requirements.txt failed; fix it and save"
            before=$(stat -c %Y requirements.txt)
            while [ "$(stat -c %Y requirements.txt)" = "$before" ]; do sleep 1; done
            continue
        fi
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
