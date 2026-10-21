"""nkpipe: end-to-end Nkhotakota camera-trap pipeline.

Stages (each is `python -m nkpipe.<stage> --config configs/pipeline.yaml`):
  download -> prepare -> train_ultralytics / train_classifier -> run_megadetector
  -> predict_ultralytics / predict_mdcls / predict_yolov4 -> evaluate -> build_sim_config
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("nkpipe")


class Paths:
    def __init__(self, cfg):
        self.work = _abs(cfg["paths"]["work"])
        self.results = _abs(cfg["paths"]["results"])
        self.raw = self.work / "raw"
        self.metadata = self.raw / "metadata"
        self.images = self.raw / "images"
        self.data = self.work / "data"
        self.manifest = self.data / "manifest.csv"
        self.boxes = self.data / "boxes.csv"
        self.classes = self.data / "classes.json"
        self.yolo = self.data / "yolo"
        self.crops = self.data / "crops"
        self.coco_test = self.data / "coco_test.json"
        self.runs = self.work / "runs"
        self.models = self.work / "models"
        self.preds = self.work / "preds"
        self.energy = self.work / "energy"
        self.eval = self.results / "eval"
        # Shared by every config (trial and full runs), so the Zenodo model is fetched once.
        self.yolov4 = _abs(cfg.get("download", {}).get("yolov4_dir", "external/yolov4_reference"))

    def mkdirs(self):
        for p in (self.metadata, self.images, self.data, self.runs, self.models,
                  self.preds, self.energy, self.eval):
            p.mkdir(parents=True, exist_ok=True)


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_cfg(path=None):
    """Load a YAML config; a `_base:` key names a config (relative to this one) to override."""
    path = Path(path or ROOT / "configs" / "pipeline.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "_base" in cfg:
        cfg = _merge(load_cfg(path.parent / cfg.pop("_base")), cfg)
    return cfg


def cli(description, extra=None):
    """Common argument parsing: returns (args, cfg, paths)."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(ROOT / "configs" / "pipeline.yaml"))
    if extra:
        extra(ap)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    paths = Paths(cfg)
    paths.mkdirs()
    return args, cfg, paths


def read_json(p):
    with open(p) as f:
        return json.load(f)


def write_json(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_classes(paths):
    return read_json(paths.classes)["names"]
