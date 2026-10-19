"""Run the published NWR_YOLO_v1 (YOLOv4, Appel et al. 2025) on the test split.

Uses OpenCV's Darknet loader, so no Darknet build is needed. Class names are
matched to this study's classes after normalisation; set
predict.yolov4.name_map for any that do not match. Detections of unmatched
classes are dropped.

CAUTION: this model was trained on Nkhotakota images whose split is not
known, so some of our test images may have been in its training data. Its
scores are therefore an optimistic reference, not a clean comparison.

    python -m nkpipe.download --yolov4 --metadata-only   # fetch the model
    python -m nkpipe.predict_yolov4
"""

import cv2
import numpy as np

from . import cli, load_classes, log, write_json
from .energy import EnergyMeter
from .prepare import norm_name
from . import preds as P

NAME = "YOLOv4_reference"


def find_files(d):
    def one(pat):
        hits = sorted(d.rglob(pat))
        return hits[0] if hits else None
    return one("*.cfg"), one("*.weights"), one("*.names") or one("*obj*.txt") or one("*classes*.txt")


def input_size(cfg_path):
    w = h = 608
    for line in open(cfg_path):
        k, _, v = line.partition("=")
        k = k.strip()
        if k == "width":
            w = int(v)
        elif k == "height":
            h = int(v)
            break
    return w, h


def main():
    args, cfg, paths = cli(__doc__)
    if not cfg["predict"]["yolov4"]["enabled"]:
        log.info("YOLOv4 reference disabled in config")
        return
    d = paths.models / "yolov4_reference"
    cfg_p, w_p, names_p = find_files(d) if d.exists() else (None, None, None)
    if not (cfg_p and w_p and names_p):
        raise SystemExit(f"YOLOv4 .cfg/.weights/.names not found in {d}; run download with --yolov4 "
                         "or place the Zenodo files there.")
    v4_names = [l.strip() for l in open(names_p) if l.strip()]
    ours = load_classes(paths)
    idx = {c: i for i, c in enumerate(ours)}
    user_map = {norm_name(k): norm_name(v) for k, v in cfg["predict"]["yolov4"]["name_map"].items()}
    mapping = {}
    for j, n in enumerate(v4_names):
        nn_ = user_map.get(norm_name(n), norm_name(n))
        if nn_ in idx:
            mapping[j] = idx[nn_]
    unmatched = [n for j, n in enumerate(v4_names) if j not in mapping]
    log.info("YOLOv4 classes matched: %d/%d; unmatched: %s", len(mapping), len(v4_names), unmatched)
    write_json({"mapping": {v4_names[j]: ours[i] for j, i in mapping.items()}, "unmatched": unmatched},
               paths.preds / "yolov4_class_mapping.json")

    net = cv2.dnn.readNetFromDarknet(str(cfg_p), str(w_p))
    use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, "cuda") else False
    if use_cuda:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        log.warning("OpenCV has no CUDA: YOLOv4 runs on CPU (slow but works)")
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=input_size(cfg_p), scale=1 / 255.0, swapRB=True)

    pc = cfg["predict"]
    test = P.test_images(paths)
    dets = {}
    with EnergyMeter(device_index=None if not use_cuda else str(cfg["train"]["device"])) as em:
        for i, r in enumerate(test.itertuples(), 1):
            img = cv2.imread(r.path)
            if img is None:
                dets[r.uid] = []
                continue
            classes, scores, boxes = model.detect(img, confThreshold=pc["conf"], nmsThreshold=0.45)
            classes = np.asarray(classes).reshape(-1).astype(int)
            scores = np.asarray(scores).reshape(-1)
            boxes = np.asarray(boxes).reshape(-1, 4)
            dets[r.uid] = [[float(b[0]), float(b[1]), float(b[2]), float(b[3]), mapping[c], float(s)]
                           for c, s, b in zip(classes, scores, boxes) if c in mapping]
            if i % 1000 == 0:
                log.info("YOLOv4: %d/%d", i, len(test))
    timing = P.timing_from_energy(em.summary(), len(test))
    timing["device"] = "cuda" if use_cuda else "cpu"
    if not use_cuda:
        timing["cpu_ms_per_image"] = timing["ms_per_image"]
    P.save(paths, NAME, ours, dets, timing)


if __name__ == "__main__":
    main()
