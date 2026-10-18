"""Fit a detector profile from per-image test-set predictions and write it
into a copy of the config.

    python scripts/calibrate.py --predictions preds_yolo11.csv --model YOLO11 \
        --infer-ms-gpu 12 --train-gpu-hours 30 --out configs/calibrated.json

See wildsim/calibrate.py for the expected CSV columns.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wildsim import load_config  # noqa: E402
from wildsim.calibrate import fit_profile  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=None, help="config to update (default: configs/default.json)")
    ap.add_argument("--infer-ms-gpu", type=float)
    ap.add_argument("--infer-ms-cpu", type=float)
    ap.add_argument("--infer-ms-edge", type=float)
    ap.add_argument("--train-gpu-hours", type=float)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = load_config(a.config)
    old = cfg["models"].get(a.model, {})
    infer = dict(old.get("infer_ms", {}))
    for k in ("gpu", "cpu", "edge"):
        v = getattr(a, f"infer_ms_{k}")
        if v is not None:
            infer[k] = v
    prof = fit_profile(pd.read_csv(a.predictions), cfg["species"], infer_ms=infer,
                       train_gpu_hours=a.train_gpu_hours or old.get("train_gpu_hours"))
    cfg["models"][a.model] = prof
    Path(a.out).write_text(json.dumps(cfg, indent=2) + "\n")
    print(json.dumps(prof, indent=2))


if __name__ == "__main__":
    main()
