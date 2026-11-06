# >>> devbox >>>
export DEVBOX_WORKSPACE="$WORKSPACE"
export DEVBOX_WORK="$WORK"
export CLAUDE_CONFIG_DIR="$CLAUDE_DIR"
export HISTFILE="$WORKSPACE/.bash_history"
export HISTSIZE=100000
[ -f "$WARNING_FILE" ] && cat "$WARNING_FILE"
[ -f "$WORKSPACE/.devboxrc" ] && . "$WORKSPACE/.devboxrc"
cd "$WORK"
# <<< devbox <<<
