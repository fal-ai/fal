# >>> devbox >>>
[ -f "$WARNING_FILE" ] && cat "$WARNING_FILE"
if [ -n "$PS1" ]; then
  printf '\e[2mrc file: ~/.devboxrc (%s/.devboxrc). /data is cold after a restart: copy weights you need fast to %s.\e[0m\n' "$DEVBOX_WORKSPACE" "$DEVBOX_WORK"
  # Default prompt: box, GPU, path, git branch. Set before .devboxrc is sourced, so
  # a PS1 set there wins.
  __devbox_gpu() {
    local cache=/tmp/devbox-gpu-short
    if [ ! -s "$cache" ]; then
      command -v nvidia-smi >/dev/null 2>&1 || return 0
      nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 |
        sed -E 's/NVIDIA|GeForce|Tesla//g; s/ (Blackwell|Server|Workstation|Max-Q|PCIe|SXM[0-9]*|HBM[0-9a-z]*|[0-9]+GB).*//; s/-.*//; s/ //g' \
        >"$cache" 2>/dev/null
    fi
    cat "$cache" 2>/dev/null
  }
  __devbox_git() {
    local b
    b=$(git symbolic-ref --short -q HEAD 2>/dev/null || git rev-parse --short HEAD 2>/dev/null) || return 0
    [ -n "$b" ] && printf ' (%s)' "$b"
  }
  __DEVBOX_GPU=$(__devbox_gpu)
  PROMPT_DIRTRIM=3
  PS1='\[\e[1;35m\]$BOX_NAME\[\e[0m\]${__DEVBOX_GPU:+ \[\e[32m\]$__DEVBOX_GPU\[\e[0m\]} \[\e[34m\]\w\[\e[0m\]\[\e[33m\]$(__devbox_git)\[\e[0m\] \$ '
  # --starship: setup() copied the local config here and installed starship.
  if [ -f "$HOME/.config/starship.toml" ] && command -v starship >/dev/null 2>&1; then
    eval "$(starship init bash)"
  fi
fi
[ -f "$WORKSPACE/.devboxrc" ] && . "$WORKSPACE/.devboxrc"
cd "$WORK"
# <<< devbox <<<
