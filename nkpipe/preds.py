"""Common prediction format shared by all models.

work/preds/<model>.json:
  {"model": name, "classes": [...],
   "detections": {uid: [[x, y, w, h, cls_idx, score], ...]},   # absolute pixels
   "timing": {"n_images", "wall_s", "ms_per_image", "gpu_kwh", "kwh_per_image", ...}}
"""

import pandas as pd

from . import read_json, write_json


def test_images(paths):
    man = pd.read_csv(paths.manifest)
    return man[man.split == "test"].reset_index(drop=True)


def save(paths, model, classes, dets, timing):
    write_json({"model": model, "classes": classes, "detections": dets, "timing": timing},
               paths.preds / f"{model}.json")


def load(paths, model):
    return read_json(paths.preds / f"{model}.json")


def timing_from_energy(summary, n_images):
    kwh = summary.get("gpu_kwh")
    return {"n_images": n_images, "wall_s": summary["wall_s"],
            "ms_per_image": 1000 * summary["wall_s"] / max(n_images, 1),
            "gpu_kwh": kwh, "kwh_per_image": kwh / max(n_images, 1) if kwh is not None else None,
            "mean_gpu_watts": summary.get("mean_gpu_watts")}
