#!/usr/bin/env bash
# Run one job on Kaggle (if compute=kaggle) else locally. Falls back to local on Kaggle failure (and records a BLOCKER).
# Usage: pipelines/run_job.sh <job_dir_name> <script> <out_dir> [local extra args...]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source .venv/bin/activate
JOB="$1"; SCRIPT="$2"; OUT="$3"; shift 3
DATA="${DATA:-data}"
COMPUTE="$(python -c "import yaml;print(yaml.safe_load(open('configs/compute.yaml'))['compute'])")"
USER_="$(python -c "import yaml;print(yaml.safe_load(open('configs/compute.yaml')).get('kaggle_user') or '')")"
mkdir -p "$OUT"
if [ "${SMOKE:-0}" != "1" ] && [ "$COMPUTE" = "kaggle" ]; then
  K="runs/train_all/kernels/$JOB"; rm -rf "$K"; mkdir -p "$K"
  cp "kaggle/jobs/$JOB/$SCRIPT" "$K/"; sed "s/__USER__/$USER_/g" "kaggle/jobs/$JOB/kernel-metadata.json" > "$K/kernel-metadata.json"
  if bash tools/kaggle_run.sh "$K" "$OUT/kaggle_out" 300 13; then
    cp -r "$OUT"/kaggle_out/* "$OUT"/ 2>/dev/null; echo "{\"compute\":\"kaggle_gpu\"}" > "$OUT/compute.json"; exit 0
  fi
  echo "[run_job] Kaggle failed for $JOB -> BLOCKER + local fallback"
  echo "- $(date '+%F %T') Kaggle job $JOB failed; reran locally (compute=local_mps)" >> BLOCKERS.md
  bash tools/notify.sh "Temple XAI: Kaggle job $JOB failed" "falling back to local run"
fi
LOCAL_ARGS="$*"
python "kaggle/jobs/$JOB/$SCRIPT" --data-root "$DATA" --out "$OUT" --device auto --compute-label local_mps $LOCAL_ARGS || exit $?
echo "{\"compute\":\"local_mps\"}" > "$OUT/compute.json"
