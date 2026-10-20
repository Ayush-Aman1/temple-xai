#!/usr/bin/env bash
# Temple XAI training pipeline. Idempotent: stages leave runs/train_all/stages/<name>.done.
# SMOKE=1 -> ~5 temples, data_smoke/, artifacts_smoke/, local only, no uploads.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT"
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then DATA=data_smoke; ART=artifacts_smoke; STG=runs/train_smoke/stages; CACHE=data/cache; else DATA=data; ART=artifacts; STG=runs/train_all/stages; CACHE=data/cache; fi
export DATA ART
mkdir -p "$STG" "$DATA" "$ART" "$ART/raw"
SM=""; [ "$SMOKE" = "1" ] && SM="--smoke"
fail() { echo "FAILED stage: $1"; bash tools/notify.sh "Temple XAI: training FAILED" "$1 — see runs/train_all/log.txt"; exit 1; }
stage() { # stage <name> <command...>
  local n="$1"; shift
  if [ -f "$STG/$n.done" ]; then echo "[skip] $n"; return 0; fi
  echo "[$(date '+%T')] >>> $n"
  "$@" || fail "$n"
  touch "$STG/$n.done"
}
COMPUTE_NOTE() { python -c "import yaml;print(yaml.safe_load(open('configs/compute.yaml'))['compute'])"; }

# ---- environment / compute decision
if [ "$SMOKE" != "1" ] && [ ! -f "$STG/env.done" ]; then
  if kaggle kernels list --mine --page-size 1 >/dev/null 2>&1; then
    U="$(python -c "from templexai.data.package_dataset import kaggle_username;print(kaggle_username() or '')")"
    printf 'compute: kaggle\nlocal_device: auto\nkaggle_user: %s\n' "$U" > configs/compute.yaml
  else
    echo "- $(date '+%F') Kaggle auth failed (see START_HERE.md) -> compute: local" >> BLOCKERS.md
    bash tools/notify.sh "Temple XAI: Kaggle unavailable" "Running locally"
    printf 'compute: local\nlocal_device: auto\nkaggle_user: null\n' > configs/compute.yaml
  fi
  touch "$STG/env.done"
fi
echo "compute=$(COMPUTE_NOTE)"

# ---- A. data
stage a0_style_map  python -m templexai.data.verify_style_map
stage a0_knowledge  python -m templexai.knowledge.compile_rules
stage a1_temples    python -m templexai.data.wikidata_temples --data-root "$DATA" $SM
stage a2_labels     python -m templexai.data.labels --data-root "$DATA" $SM
stage a3_images     python -m templexai.data.commons_images --data-root "$DATA" $SM
stage a4_imagetype  python -m templexai.data.tag_image_type --data-root "$DATA" $SM
stage a5_split      python -m templexai.data.make_splits --data-root "$DATA" $SM
if [ "$SMOKE" != "1" ]; then
  stage a6_package  python -m templexai.data.package_dataset --data-root "$DATA"
  if [ "$(COMPUTE_NOTE)" = "kaggle" ]; then stage a6_upload bash tools/kaggle_dataset.sh kaggle/dataset; fi
fi

# ---- B. jobs
RAW="$ART/raw"
if [ "$SMOKE" = "1" ]; then
  stage j0 bash pipelines/run_job.sh j0_gpu_check gpu_check.py "$RAW/j0_gpu_check"
  stage j1 bash pipelines/run_job.sh j1_zeroshot zeroshot.py "$RAW/j1_zeroshot" --smoke
  mkdir -p "$DATA/derived"; cp "$RAW/j1_zeroshot/boxes_zeroshot.json" "$DATA/derived/" 2>/dev/null
  stage j2 bash pipelines/run_job.sh j2_classifiers classifiers.py "$RAW/j2_classifiers" --smoke --epochs 1 --seeds 0 --img-size 160 --workers 0
  stage j3 bash pipelines/run_job.sh j3_detector detector.py "$RAW/j3_detector" --smoke
else
  stage j0 bash pipelines/run_job.sh j0_gpu_check gpu_check.py "$RAW/j0_gpu_check"
  if [ "$(COMPUTE_NOTE)" = "kaggle" ] && ! grep -q '"cuda_available": true' "$RAW/j0_gpu_check/gpu.json" 2>/dev/null; then
    echo "- $(date '+%F') Kaggle GPU unavailable (phone verification?) -> switching remaining jobs to local" >> BLOCKERS.md
    bash tools/notify.sh "Temple XAI: Kaggle GPU unavailable" "phone verification? switching to local"
    sed -i.bak 's/compute: kaggle/compute: local/' configs/compute.yaml; rm -f configs/compute.yaml.bak
  fi
  stage j1 bash pipelines/run_job.sh j1_zeroshot zeroshot.py "$RAW/j1_zeroshot"
  mkdir -p kaggle/dataset_boxes "$DATA/derived"
  cp "$RAW/j1_zeroshot/boxes_zeroshot.json" kaggle/dataset_boxes/; cp "$RAW/j1_zeroshot/boxes_zeroshot.json" "$DATA/derived/"
  if [ "$(COMPUTE_NOTE)" = "kaggle" ]; then
    U="$(python -c "import yaml;print(yaml.safe_load(open('configs/compute.yaml'))['kaggle_user'])")"
    printf '{"title":"templexai-boxes","id":"%s/templexai-boxes","licenses":[{"name":"other"}],"isPrivate":true,"subtitle":"Temple XAI zero-shot component boxes (J1 output)"}' "$U" > kaggle/dataset_boxes/dataset-metadata.json
    stage a6b_upload_boxes bash tools/kaggle_dataset.sh kaggle/dataset_boxes   # separate small dataset: avoids re-uploading the 2 GB images
  fi
  stage j2 bash pipelines/run_job.sh j2_classifiers classifiers.py "$RAW/j2_classifiers" --epochs 3 --img-size 192 --seeds 0 1 2 --workers 0
  stage j3 bash pipelines/run_job.sh j3_detector detector.py "$RAW/j3_detector" --epochs 10 --imgsz 416
fi
stage collect python -m templexai.artifacts.collect --data-root "$DATA" --art "$ART" --raw "$RAW"
echo "ALL STAGES DONE"
[ "$SMOKE" = "1" ] || bash tools/notify.sh "Temple XAI: training DONE" "Next: prompts/02_BUILD.md"
