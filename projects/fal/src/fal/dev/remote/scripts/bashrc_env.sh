# >>> devbox-env >>>
# Sourced from the top of ~/.bashrc, before the stock "not interactive" early return,
# so ssh commands, editor tasks and scripts get the box environment too. Only
# variables and functions: nothing here may print, or scp and rsync break.
# Your secrets and variables (--secret, --set-env); never the watchdog's key.
[ -r "$USER_ENV_FILE" ] && . "$USER_ENV_FILE"
unset $ADMIN_KEY
export DEVBOX_WORKSPACE="$WORKSPACE"
export DEVBOX_WORK="$WORK"
export CLAUDE_CONFIG_DIR="$CLAUDE_DIR"
export HISTFILE="$WORKSPACE/.bash_history"
export HISTSIZE=100000
__devbox_data_note() {
  case "$1/" in
    /data/*)
      printf '\e[33mnote:\e[0m %s is on the /data network volume, so environments there are slow.\n      Keep them in %s (local disk, saved with the box) and put models and data on /data.\n' "$1" "$DEVBOX_WORK" >&2
      ;;
  esac
}
# The first argument that is not a flag or a flag's value, made absolute; empty if
# there is none.
__devbox_first_path() {
  local arg path= skip=
  for arg; do
    if [ -n "$skip" ]; then skip=; continue; fi
    case "$arg" in
      -p | --python | --python-preference | --prompt | --seed) skip=1 ;;
      -*) ;;
      *) path=$arg; break ;;
    esac
  done
  case "$path" in
    "" | /*) printf '%s' "$path" ;;
    *) printf '%s/%s' "$PWD" "$path" ;;
  esac
}
__devbox_asks_help() {
  local arg
  for arg; do
    case "$arg" in -h | --help) return 0 ;; esac
  done
  return 1
}
uv() {
  if ! __devbox_asks_help "$@"; then
    case "${1-}" in
      venv)
        local dir
        dir=$(__devbox_first_path "${@:2}")
        __devbox_data_note "${dir:-$PWD}"
        ;;
      sync | run) __devbox_data_note "${UV_PROJECT_ENVIRONMENT:-$PWD}" ;;
      pip)
        case "${2-}" in
          list | show | freeze | check | tree) ;;
          *) __devbox_data_note "${VIRTUAL_ENV:-$PWD}" ;;
        esac
        ;;
    esac
  fi
  command uv "$@"
}
command -v python3 >/dev/null 2>&1 && python3() {
  if [ "${1-}" = -m ] && [ "${2-}" = venv ] && ! __devbox_asks_help "$@"; then
    local dir
    dir=$(__devbox_first_path "${@:3}")
    __devbox_data_note "${dir:-$PWD}"
  fi
  command python3 "$@"
}
command -v python >/dev/null 2>&1 && python() {
  if [ "${1-}" = -m ] && [ "${2-}" = venv ] && ! __devbox_asks_help "$@"; then
    local dir
    dir=$(__devbox_first_path "${@:3}")
    __devbox_data_note "${dir:-$PWD}"
  fi
  command python "$@"
}
# <<< devbox-env <<<
