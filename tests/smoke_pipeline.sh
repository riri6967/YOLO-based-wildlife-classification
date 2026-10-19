#!/usr/bin/env bash
# End-to-end smoke test on synthetic data, CPU only (~10-20 min).
# Exercises every stage except the real download, MegaDetector (faked) and YOLOv4.
#   bash tests/smoke_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH || true
PY=${PY:-.venv/bin/python}
[[ -x "$PY" ]] || PY=python3
C=tests/smoke.yaml
rm -rf work_smoke results_smoke
$PY tests/make_synthetic_lila.py --work work_smoke
$PY -m nkpipe.prepare --config $C
$PY -m nkpipe.train_ultralytics --config $C --model yolo11
$PY -m nkpipe.train_ultralytics --config $C --model rtdetr
$PY -m nkpipe.train_classifier --config $C
$PY tests/fake_megadetector.py $C
$PY -m nkpipe.predict_ultralytics --config $C --model YOLO11
$PY -m nkpipe.predict_ultralytics --config $C --model RT-DETR
$PY -m nkpipe.predict_mdcls --config $C
$PY -m nkpipe.evaluate --config $C
$PY -m nkpipe.build_sim_config --config $C
$PY scripts/run_experiment.py --config results_smoke/simulation/calibrated_config.json --reps 1 \
    --out results_smoke/simulation
$PY scripts/run_sensitivity.py --config results_smoke/simulation/calibrated_config.json --n-base 4 \
    --out results_smoke/simulation
echo "SMOKE TEST PASSED"
