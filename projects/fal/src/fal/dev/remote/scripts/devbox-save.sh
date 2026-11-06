#!/bin/bash
# Saves $WORK to $SNAPSHOT.
# The box runs it every $INTERVAL min and before it stops; `fal dev down` runs it
# too. --force saves even without changes.
# box_app.py fills in the capitalized names when it installs this script.
set -euo pipefail
exec 9>/tmp/devbox-save.lock
flock 9
stamp=$STAMP
me=$(hostname)
if [ "${1:-}" != --force ] && [ -f "$stamp" ] \
        && [ -z "$(find $WORK -newer "$stamp" -print -quit)" ]; then
    echo "devbox-save: no changes since the last save"
    exit 0
fi
start=$(date +%s)
touch "$stamp.next"
tmp=$SNAPSHOT.$me.tmp
# tar exits 1 when a file changes while it is read; the next save picks that up.
{ tar -C $WORK_PARENT -cf - $WORK_NAME || [ $? -eq 1 ]; } \
    | zstd -q -1 -T0 -f -o "$tmp"
# Only the lease owner replaces the snapshot; a newer runner has taken over otherwise.
owner=$(cut -d' ' -f1 $LEASE 2>/dev/null || true)
if [ -n "$owner" ] && [ "$owner" != "$me" ]; then
    kept=$SNAPSHOT.orphan-$me-$(date +%s)
    mv -f "$tmp" "$kept"
    echo "devbox-save: runner $owner now owns $WORK;" \
        "this copy was kept at $kept" >&2
    exit 1
fi
mv -f "$tmp" $SNAPSHOT
mv -f "$stamp.next" "$stamp"
size=$(du -h $SNAPSHOT | cut -f1)
echo "devbox-save: saved $WORK ($size) in $(( $(date +%s) - start ))s"
