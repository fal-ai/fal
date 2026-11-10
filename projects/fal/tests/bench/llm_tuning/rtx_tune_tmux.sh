#!/bin/sh
# One tmux window for the RTX tuning loop: Claude on the left, the box dash top
# right, the local proxy bottom right.
#
#   sh rtx_tune_tmux.sh [box-name]
#
# The box must already be started (`fal dev app start ...`). This script only
# attaches the three panes to it, so it costs nothing on its own.

set -eu

BOX="${1:-qwen38tune}"
# Three levels up from tests/bench/llm_tuning lands on projects/fal, so this
# works from any checkout or worktree rather than one person's layout.
FAL_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK_DIR="${RTX_TUNE_WORK_DIR:-$PWD}"
SESSION="rtxtune"
FAL="uv run --directory $FAL_DIR --python 3.12 fal"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  exec tmux attach -t "$SESSION"
fi

# Left pane: Claude, driving the loop.
tmux new-session -d -s "$SESSION" -c "$WORK_DIR" -x "$(tput cols)" -y "$(tput lines)"
tmux send-keys -t "$SESSION" "claude" C-m

# Right column, top: the live dash for the box.
tmux split-window -h -t "$SESSION" -c "$WORK_DIR" -p 45
tmux send-keys -t "$SESSION" "$FAL dev app dash --name $BOX" C-m

# Right column, bottom: the local proxy the bench talks to.
tmux split-window -v -t "$SESSION" -c "$WORK_DIR" -p 40
tmux send-keys -t "$SESSION" "$FAL dev app proxy --name $BOX" C-m

tmux select-pane -t "$SESSION".0
exec tmux attach -t "$SESSION"
