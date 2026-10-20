#!/usr/bin/env bash
# Create (first time) or version a PRIVATE Kaggle dataset from a folder containing
# dataset-metadata.json, then wait until Kaggle reports it ready.
# Usage: tools/kaggle_dataset.sh <dataset_dir>
# Exit: 0 ready | 2 upload failed | 3 processing error | 4 timed out waiting
set -uo pipefail
DDIR="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.venv/bin/activate" ] && source "$ROOT/.venv/bin/activate"
ID="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['id'])" "$DDIR/dataset-metadata.json")"
if kaggle datasets files "$ID" >/dev/null 2>&1; then
  echo "[kaggle_dataset] new version of $ID"
  kaggle datasets version -p "$DDIR" -m "update $(date '+%F %H:%M')" --dir-mode zip || exit 2
else
  echo "[kaggle_dataset] creating private dataset $ID"
  kaggle datasets create -p "$DDIR" --dir-mode zip || exit 2
fi
# readiness: `datasets status` is forbidden for some access tokens (403), so poll `datasets files` instead
for _ in $(seq 1 120); do
  if kaggle datasets files "$ID" --page-size 2000 2>/dev/null | grep -q "images/\|boxes_zeroshot"; then echo "[kaggle_dataset] $(date '+%T') ready (files listed)"; exit 0; fi
  echo "[kaggle_dataset] $(date '+%T') waiting for processing"
  sleep 30
done
exit 4
