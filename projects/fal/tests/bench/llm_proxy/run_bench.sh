#!/bin/bash
# Usage: run_bench.sh <label> <proxy extra args...>  (env: STREAMS="1 64 256", TOK_RATE, TPC)
# Starts bench engine (:18000) and proxy (:18001), drives load direct and via proxy,
# and samples proxy CPU with ps.
set -u
ulimit -n 65536
HERE="$(dirname "$0")"
PY="${PY:-python}"
LABEL="$1"; shift
$PY "$HERE/bench_engine.py" --port 18000 --procs "${ENGINE_PROCS:-4}" &
ENG=$!
sleep 1
$PY "$HERE/bench_proxy.py" --port 18001 --engine-port 18000 "$@" > "${OUT:-/tmp}/proxy-$LABEL.log" 2>&1 &
PROXY=$!
for i in $(seq 1 60); do
  curl -sf localhost:18001/health >/dev/null && break
  sleep 0.5
done
for s in ${STREAMS:-1 64 256 512 1024}; do
  for target in 18000 18001; do
    t0=$(ps -o cputime= -p $PROXY)
    $PY "$HERE/load.py" --port $target --streams $s --tok-rate "${TOK_RATE:-50}" --tpc "${TPC:-1}" \
      --duration "${DURATION:-12}" --procs "${LOAD_PROCS:-8}" --label "$LABEL:$([ $target = 18000 ] && echo direct || echo proxy)"
    t1=$(ps -o cputime= -p $PROXY)
    echo "  proxy cputime $t0 -> $t1"
  done
done
[ -n "${PROF:-}" ] && { kill -USR1 $PROXY; sleep 3; }
kill $PROXY 2>/dev/null
pkill -P $ENG 2>/dev/null
kill $ENG 2>/dev/null
wait 2>/dev/null
