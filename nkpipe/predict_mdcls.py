"""Stage 2 of the two-stage pipeline: classify MegaDetector animal crops.

Detection score = MegaDetector confidence x classifier probability, so a
single threshold acts on both stages (comparable with one-stage detectors).

    python -m nkpipe.predict_mdcls
"""

import torch
from PIL import Image

from . import cli, log, read_json
from .energy import EnergyMeter
from .train_classifier import build_model, eval_transform
from . import preds as P

NAME = "MegaDetector+classifier"


def crop_square(img, box_norm, margin):
    W, H = img.size
    x, y, w, h = box_norm[0] * W, box_norm[1] * H, box_norm[2] * W, box_norm[3] * H
    side = max(w, h) * (1 + 2 * margin)
    cx, cy = x + w / 2, y + h / 2
    return img.crop((max(0, cx - side / 2), max(0, cy - side / 2),
                     min(W, cx + side / 2), min(H, cy + side / 2))), (x, y, w, h)


def main():
    args, cfg, paths = cli(__doc__)
    mc, pc = cfg["megadetector"], cfg["prepare"]
    md = read_json(paths.preds / "megadetector_raw.json")
    animal_ids = {k for k, v in md.get("detection_categories", {"1": "animal"}).items() if v == "animal"}
    by_path = {im["file"]: im for im in md["images"]}

    ck = torch.load(paths.models / "classifier.pt", map_location="cpu")
    names = ck["names"]
    dev = torch.device("cuda" if torch.cuda.is_available() and cfg["train"]["device"] != "cpu" else "cpu")
    model = build_model(ck["arch"], len(names), False)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(dev)
    tf = eval_transform(ck["img_size"])

    test = P.test_images(paths)
    dets = {}
    with EnergyMeter(device_index=None if dev.type == "cpu" else str(cfg["train"]["device"])) as em:
        for i, r in enumerate(test.itertuples(), 1):
            if i % 200 == 0:
                log.info("classified crops for %d/%d images", i, len(test))
            im = by_path.get(r.path) or by_path.get(str(r.path))
            dets[r.uid] = []
            if not im or not im.get("detections"):
                continue
            boxes = [d for d in im["detections"] if d["category"] in animal_ids
                     and d["conf"] >= mc["min_conf_for_classifier"]]
            boxes = sorted(boxes, key=lambda d: -d["conf"])[: mc["max_dets_per_image"]]
            if not boxes:
                continue
            img = Image.open(r.path).convert("RGB")
            crops, abs_boxes = zip(*(crop_square(img, d["bbox"], pc["crop_margin"]) for d in boxes))
            x = torch.stack([tf(c) for c in crops]).to(dev)
            with torch.no_grad():
                prob = model(x).softmax(1).cpu()
            pmax, cls = prob.max(1)
            dets[r.uid] = [[*b, int(c), float(d["conf"] * p)]
                           for b, c, p, d in zip(abs_boxes, cls.tolist(), pmax.tolist(), boxes)]
    cls_t = P.timing_from_energy(em.summary(), len(test))
    md_e = read_json(paths.energy / "MegaDetector_infer.json")
    md_t = P.timing_from_energy(md_e, len(test))
    timing = {
        "n_images": len(test),
        "wall_s": md_t["wall_s"] + cls_t["wall_s"],
        "ms_per_image": md_t["ms_per_image"] + cls_t["ms_per_image"],
        "gpu_kwh": (md_t["gpu_kwh"] or 0) + (cls_t["gpu_kwh"] or 0) if md_t["gpu_kwh"] is not None else None,
        "stage_ms_per_image": {"megadetector": md_t["ms_per_image"], "classifier": cls_t["ms_per_image"]},
    }
    timing["kwh_per_image"] = timing["gpu_kwh"] / len(test) if timing["gpu_kwh"] is not None else None
    P.save(paths, NAME, names, dets, timing)
    log.info("%s: %d test images, %.1f ms/image", NAME, len(test), timing["ms_per_image"])


if __name__ == "__main__":
    main()
