"""Create a tiny fake LILA/COCO-Camera-Traps dataset for smoke-testing the pipeline.

Coloured shapes stand in for animals; some frames are greyscale "IR" night
images, some are low-contrast "rain" frames, and sites differ in how green
they are. Writes <work>/raw/metadata/merged.json and the images.

    python tests/make_synthetic_lila.py --work /tmp/nk_smoke/work
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SPECIES = {"elephant": (90, 90, 90), "sable_antelope": (60, 30, 10), "warthog": (150, 110, 70),
           "genet": (230, 220, 200), "leopard": (220, 170, 40)}


def make_image(rng, species, green, night, rain, W=320, H=240):
    base = np.array([60 + 40 * (1 - green), 90 + 80 * green, 50], float)
    a = np.clip(base + rng.normal(0, 25, (H, W, 3)), 0, 255)
    boxes = []
    img = Image.fromarray(a.astype(np.uint8))
    d = ImageDraw.Draw(img)
    for sp in species:
        w, h = rng.integers(40, 110), rng.integers(30, 80)
        x, y = rng.integers(0, W - w), rng.integers(30, H - h - 20)
        col = SPECIES[sp]
        (d.ellipse if sp in ("elephant", "genet") else d.rectangle)([x, y, x + w, y + h], fill=col)
        boxes.append((sp, [float(x), float(y), float(w), float(h)]))
    if night:
        img = img.convert("L").convert("RGB")
    if rain:
        arr = np.asarray(img, float)
        img = Image.fromarray((arr * 0.3 + 140).astype(np.uint8))
    return img, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--per-site", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rnd = random.Random(a.seed)
    work = Path(a.work)
    img_dir, meta_dir = work / "raw" / "images", work / "raw" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    cats = [{"id": 0, "name": "empty"}, {"id": 99, "name": "human"}] + \
           [{"id": i + 1, "name": s.replace("_", " ")} for i, s in enumerate(SPECIES)]
    cid = {c["name"].replace(" ", "_"): c["id"] for c in cats}
    images, anns = [], []
    sites = [f"{r}{c:02d}" for r in "BCDEFGHI" for c in (3, 9)]
    for si, site in enumerate(sites):
        green = rng.random()
        t0 = datetime(2020, 1 + (si % 12), 1, 0, 0)
        for k in range(a.per_site):
            t = t0 + timedelta(hours=int(rng.integers(0, 24 * 60)), minutes=int(rng.integers(0, 60)))
            night = t.hour >= 18 or t.hour < 6
            rain = rng.random() < 0.1
            r = rng.random()
            if r < 0.25:
                sp = []
            else:
                pool = list(SPECIES)
                if site[0] in "BC":
                    pool.remove("leopard")
                sp = [rnd.choice(pool)] * (1 + int(rng.random() < 0.15))
            img, boxes = make_image(rng, sp, green, night, rain)
            folder = f"{site}/{site}-A-2020-08-22"
            fn = f"{folder}/{site}-A-2020-08-22_({k}).JPG"
            (img_dir / folder).mkdir(parents=True, exist_ok=True)
            img.save(img_dir / fn, quality=90)
            iid = f"{site}_{k}"
            images.append({"id": iid, "file_name": fn, "location": site, "width": 320, "height": 240,
                           "datetime": t.strftime("%Y-%m-%d %H:%M:%S"), "seq_id": f"{site}_{k // 3}"})
            if not sp:
                anns.append({"id": f"a{len(anns)}", "image_id": iid, "category_id": 0})
            elif rng.random() < 0.15:  # image-level label only (no boxes)
                anns.append({"id": f"a{len(anns)}", "image_id": iid, "category_id": cid[sp[0]]})
            else:
                for s, b in boxes:
                    anns.append({"id": f"a{len(anns)}", "image_id": iid, "category_id": cid[s], "bbox": b})
    json.dump({"images": images, "categories": cats, "annotations": anns},
              open(meta_dir / "merged.json", "w"))
    print(f"synthetic dataset: {len(images)} images, {len(anns)} annotations in {work}")


if __name__ == "__main__":
    main()
