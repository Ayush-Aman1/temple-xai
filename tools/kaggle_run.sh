#!/usr/bin/env bash
# Push a Kaggle kernel folder (kernel-metadata.json + script), wait for it, download outputs.
# Call it from inside a background pipeline (tools/bg.sh), never as a blocking step in the agent session.
# Usage: tools/kaggle_run.sh <kernel_dir> <output_dir> [poll_seconds=300] [timeout_hours=13]
# Exit: 0 complete | 2 push failed | 3 kernel error/cancelled | 4 timeout
set -uo pipefail
KDIR="$1"; OUT="$2"; POLL="${3:-300}"; TIMEOUT_H="${4:-13}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.venv/bin/activate" ] && source "$ROOT/.venv/bin/activate"
ID="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['id'])" "$KDIR/kernel-metadata.json")"
SLUG="${ID#*/}"
mkdir -p "$OUT"
echo "[kaggle_run] $(date '+%F %T') push $ID"
if ! kaggle kernels push -p "$KDIR"; then echo "[kaggle_run] push failed"; exit 2; fi
DEADLINE=$(( $(date +%s) + TIMEOUT_H * 3600 ))
sleep 45
while :; do
  RAW="$(kaggle kernels status "$ID" 2>&1)"
  ST="$(printf '%s\n' "$RAW" | grep -i 'status' | grep -i "$SLUG" | tail -n 1 | tr '[:upper:]' '[:lower:]')"
  echo "[kaggle_run] $(date '+%F %T') ${ST:-no status line (transient?)}"
  case "$ST" in
    *complete*) break ;;
    *error*|*cancel*)
      echo "[kaggle_run] kernel ended badly; fetching whatever output/log exists"
      kaggle kernels output "$ID" -p "$OUT" >/dev/null 2>&1 || true
      exit 3 ;;
  esac
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then echo "[kaggle_run] timeout after ${TIMEOUT_H}h"; exit 4; fi
  sleep "$POLL"
done
kaggle kernels output "$ID" -p "$OUT"
echo "[kaggle_run] $(date '+%F %T') outputs saved to $OUT"
