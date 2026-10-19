"""Stand-in for MegaDetector in smoke tests: writes MegaDetector-format output
for the test split from ground-truth boxes plus jitter, misses and false boxes."""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from nkpipe import Paths, load_cfg, write_json  # noqa: E402


def main(config):
    cfg = load_cfg(config)
    paths = Paths(cfg)
    rng = np.random.default_rng(0)
    man = pd.read_csv(paths.manifest)
    test = man[man.split == "test"]
    bx = pd.read_csv(paths.boxes)
    out = []
    for r in test.itertuples():
        dets = []
        for b in bx[bx.uid == r.uid].itertuples():
            if rng.random() < 0.9:
                j = rng.normal(0, 3, 4)
                dets.append({"category": "1", "conf": float(rng.uniform(0.5, 0.99)),
                             "bbox": [(b.x + j[0]) / r.width, (b.y + j[1]) / r.height,
                                      max(b.w + j[2], 2) / r.width, max(b.h + j[3], 2) / r.height]})
        if rng.random() < 0.1:
            dets.append({"category": "1", "conf": float(rng.uniform(0.05, 0.4)), "bbox": [0.1, 0.1, 0.2, 0.2]})
        out.append({"file": r.path, "detections": dets})
    write_json({"images": out, "detection_categories": {"1": "animal", "2": "person", "3": "vehicle"}},
               paths.preds / "megadetector_raw.json")
    write_json({"wall_s": 1.0, "gpu_hours": 0.0, "gpu_kwh": None, "n_images": len(test)},
               paths.energy / "MegaDetector_infer.json")


if __name__ == "__main__":
    main(sys.argv[1])
