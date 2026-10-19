#!/usr/bin/env bash
# Run the whole study, A to Z.
#
#   bash run_all.sh                       # full run with configs/pipeline.yaml
#   bash run_all.sh --quick               # trial run (configs/quick.yaml: ~3k images, few epochs)
#   bash run_all.sh --from evaluate       # resume from a stage
#   bash run_all.sh --only train_yolo11   # run a single stage
#
# Stages: download prepare train_yolo11 train_rtdetr train_classifier megadetector
#         predict evaluate simulate
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONPATH || true   # keep ROS 2 Jazzy's Python packages out of this environment

CONFIG=configs/pipeline.yaml
FROM=""
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) CONFIG=configs/quick.yaml ;;
    --config) CONFIG="$2"; shift ;;
    --from) FROM="$2"; shift ;;
    --only) ONLY="$2"; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done

PY=${PY:-.venv/bin/python}
[[ -x "$PY" ]] || PY=python3
cfgget() { $PY -c "from nkpipe import load_cfg,_abs;c=load_cfg('$CONFIG');print($1)"; }
RESULTS=$(cfgget "_abs(c['paths']['results'])")
mkdir -p "$RESULTS"
LOG="$RESULTS/run_all.log"

started=$([[ -z "$FROM" ]] && echo 1 || echo 0)
run() {
  local stage=$1; shift
  if [[ -n "$ONLY" && "$ONLY" != "$stage" ]]; then return; fi
  if [[ "$started" == 0 && "$stage" == "$FROM" ]]; then started=1; fi
  if [[ -z "$ONLY" && "$started" == 0 ]]; then return; fi
  echo "=================== $stage ===================" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

run download          $PY -m nkpipe.download --config "$CONFIG" --yolov4
run prepare           $PY -m nkpipe.prepare --config "$CONFIG"
run train_yolo11      $PY -m nkpipe.train_ultralytics --config "$CONFIG" --model yolo11
run train_rtdetr      $PY -m nkpipe.train_ultralytics --config "$CONFIG" --model rtdetr
run train_classifier  $PY -m nkpipe.train_classifier --config "$CONFIG"
run megadetector      $PY -m nkpipe.run_megadetector --config "$CONFIG"
run predict bash -c "
  $PY -m nkpipe.predict_ultralytics --config '$CONFIG' --model YOLO11 &&
  $PY -m nkpipe.predict_ultralytics --config '$CONFIG' --model RT-DETR &&
  $PY -m nkpipe.predict_mdcls --config '$CONFIG' &&
  ( $PY -m nkpipe.predict_yolov4 --config '$CONFIG' || echo 'YOLOv4 reference skipped (see message above)' )"
run evaluate          $PY -m nkpipe.evaluate --config "$CONFIG"
SIM="$RESULTS/simulation"
run simulate bash -c "
  $PY -m nkpipe.build_sim_config --config '$CONFIG' &&
  $PY scripts/run_experiment.py --config '$SIM/calibrated_config.json' --reps $(cfgget "c['simulate']['reps']") --out '$SIM' &&
  $PY scripts/run_sensitivity.py --config '$SIM/calibrated_config.json' --n-base $(cfgget "c['simulate']['sobol_n_base']") --out '$SIM'"

echo "All done. Results in $RESULTS (log: $LOG)"
