"""Build the modelling dataset from the downloaded LILA metadata and images.

Outputs (under work/data):
  manifest.csv   one row per image: split, label, location, time, environment covariates
  boxes.csv      one row per ground-truth box (absolute pixel xywh)
  classes.json   class names and training counts
  yolo/          Ultralytics dataset (images symlinked, YOLO txt labels, data.yaml)
  coco_test.json COCO ground truth for the test split (used for mAP)
  crops/         species crops for training the MegaDetector second-stage classifier

    python -m nkpipe.prepare --config configs/pipeline.yaml
"""

import os
import random
import re
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from . import cli, log, read_json, write_json
from .download import location_of, relative_name, yolov4_seen_names
from .envfeatures import _features_star

SPLITS = ("train", "val", "test")


def norm_name(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def uid_of(rel):
    return str(Path(rel).with_suffix("")).replace("/", "__").replace("\\", "__")


# --------------------------------------------------------------------------- split

def split_locations(loc_counts, fractions, iters, seed):
    """Assign whole camera sites to train/val/test.

    loc_counts: DataFrame (location x class) of box-annotated image counts.
    Random restarts; keep the assignment that best covers every class in val
    and test while matching the target image fractions.
    """
    rng = random.Random(seed)
    locs = list(loc_counts.index)
    tot = loc_counts.values.sum()
    best, best_score = None, np.inf
    present = loc_counts.columns[(loc_counts > 0).sum() >= 2]  # classes seen at >= 2 sites
    for _ in range(iters):
        rng.shuffle(locs)
        assign, acc = {}, {"test": 0, "val": 0}
        for loc in locs:
            n = loc_counts.loc[loc].sum()
            for s in ("test", "val"):
                if acc[s] < fractions[s] * tot:
                    assign[loc] = s
                    acc[s] += n
                    break
            else:
                assign[loc] = "train"
        sp = pd.Series(assign)
        score = 0.0
        for s in SPLITS:
            sub = loc_counts.loc[sp[sp == s].index]
            frac = sub.values.sum() / tot
            score += 100 * abs(frac - fractions[s])
            missing = (sub[present].sum() == 0).sum()
            score += {"train": 50, "val": 3, "test": 10}[s] * missing
        if score < best_score:
            best, best_score = dict(assign), score
    return best


# --------------------------------------------------------------------------- helpers

def _parse_dt(v):
    try:
        return pd.to_datetime(v)
    except Exception:
        return pd.NaT


def _link(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        import shutil
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def _crop_job(args):
    path, uid, boxes, out_dir, size, margin = args
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return 0
    W, H = img.size
    n = 0
    for k, (cls, x, y, w, h) in enumerate(boxes):
        side = max(w, h) * (1 + 2 * margin)
        cx, cy = x + w / 2, y + h / 2
        box = (max(0, cx - side / 2), max(0, cy - side / 2), min(W, cx + side / 2), min(H, cy + side / 2))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            continue
        d = Path(out_dir) / str(cls)
        d.mkdir(parents=True, exist_ok=True)
        img.crop(box).resize((size, size)).save(d / f"{uid}_{k}.jpg", quality=90)
        n += 1
    return n


# --------------------------------------------------------------------------- main

def main():
    args, cfg, paths = cli(__doc__, lambda ap: ap.add_argument(
        "--force", action="store_true", help="rebuild the dataset even if it was already prepared"))
    if (paths.data / ".complete").exists() and not args.force:
        log.info("dataset already prepared in %s; skipping. Use --force to rebuild.", paths.data)
        return
    (paths.data / ".complete").unlink(missing_ok=True)
    pc, dl = cfg["prepare"], cfg["download"]
    meta = read_json(paths.metadata / "merged.json")
    cats = {c["id"]: norm_name(c["name"]) for c in meta["categories"]}
    exclude = {norm_name(c) for c in pc["exclude_categories"]}

    sel_path = paths.raw / "selected_ids.json"
    selected = set(map(str, read_json(sel_path))) if sel_path.exists() else None

    anns = defaultdict(list)
    for a in meta["annotations"]:
        anns[str(a["image_id"])].append(a)

    rows, box_rows = [], []
    for im in meta["images"]:
        iid = str(im["id"])
        if selected is not None and iid not in selected:
            continue
        rel = relative_name(im["file_name"], dl)
        path = paths.images / rel
        if not path.exists():
            continue
        a_im = anns.get(iid, [])
        boxes = [a for a in a_im if a.get("bbox")]
        names = {cats[a["category_id"]] for a in a_im}
        if boxes:
            if any(cats[a["category_id"]] in exclude for a in boxes):
                continue
            label_counts = Counter(cats[a["category_id"]] for a in boxes)
            label, kind = label_counts.most_common(1)[0][0], "boxed"
        elif names and names <= {"empty"}:
            label, kind = "empty", "empty"
        elif len(names) == 1 and not names & exclude:
            label, kind = next(iter(names)), "imagelevel"  # independent test only (no boxes)
        else:
            continue
        uid = uid_of(rel)
        rows.append({"uid": uid, "lila_id": iid, "rel": rel, "path": str(path),
                     "location": location_of(im), "meta_datetime": im.get("datetime"),
                     "seq_id": im.get("seq_id"), "label": label, "kind": kind, "n_boxes": len(boxes),
                     "meta_w": im.get("width"), "meta_h": im.get("height")})
        for a in boxes:
            x, y, w, h = a["bbox"]
            box_rows.append({"uid": uid, "cls_name": cats[a["category_id"]], "x": x, "y": y, "w": w, "h": h})

    man = pd.DataFrame(rows)
    bx = pd.DataFrame(box_rows, columns=["uid", "cls_name", "x", "y", "w", "h"])
    log.info("usable images: %d (%d with boxes, %d empty, %d image-level test)", len(man),
             (man.kind == "boxed").sum(), (man.kind == "empty").sum(), (man.kind == "imagelevel").sum())
    seen = yolov4_seen_names(paths)
    man["yolov4_seen"] = man.rel.map(lambda r: int(r.rsplit("/", 1)[-1] in seen))

    # ---- environmental covariates from pixels / EXIF
    log.info("computing image covariates with %d workers ...", pc["workers"])
    with Pool(pc["workers"]) as pool:
        feats = pool.map(_features_star, [(p, pc["ir_chroma_threshold"]) for p in man.path], chunksize=64)
    man = pd.concat([man, pd.DataFrame(feats)], axis=1)
    man = man[man.readable].reset_index(drop=True)
    man["width"] = man.width.fillna(man.meta_w)
    man["height"] = man.height.fillna(man.meta_h)
    dt = man.meta_datetime.where(man.meta_datetime.notna(), man.exif_datetime).map(_parse_dt)
    man["datetime"] = dt
    man["hour"] = dt.dt.hour
    man["month"] = dt.dt.month
    night_start, night_end = pc["night_hours"]
    clock_night = (man.hour >= night_start) | (man.hour < night_end)
    man["night"] = np.where(man.hour.notna(), clock_night | (man.ir == 1), man.ir == 1).astype(int)
    man["season"] = np.where(man.month.isin([11, 12, 1, 2, 3, 4]), "wet",
                             np.where(man.month.notna(), "dry", "unknown"))

    # Site vegetation density: median daytime green fraction per camera site, scaled 0-1.
    site_green = man[man.ir == 0].groupby("location").green_frac.median()
    lo, hi = site_green.quantile(0.02), site_green.quantile(0.98)
    site_veg = ((site_green - lo) / max(hi - lo, 1e-6)).clip(0, 1)
    man["vegetation"] = man.location.map(site_veg).fillna(site_veg.median() if len(site_veg) else 0.5)

    # Poor visibility proxy: lowest contrast or sharpness decile within IR / colour frames.
    q = pc["poor_visibility_quantile"]
    man["poor_visibility"] = 0
    for ir_val, g in man.groupby("ir"):
        bad = (g.contrast < g.contrast.quantile(q)) | (g.sharpness < g.sharpness.quantile(q))
        man.loc[g.index[bad], "poor_visibility"] = 1

    # ---- classes
    bx = bx[bx.uid.isin(man.uid)]
    img_per_cls = bx.drop_duplicates(["uid", "cls_name"]).cls_name.value_counts()
    keep = sorted(img_per_cls[img_per_cls >= pc["min_images_per_class"]].index)
    dropped = sorted(set(img_per_cls.index) - set(keep))
    if dropped:
        bad_uids = set(bx[bx.cls_name.isin(dropped)].uid)
        log.info("dropping %d rare classes (%s) and their %d images", len(dropped), ", ".join(dropped), len(bad_uids))
        man = man[~man.uid.isin(bad_uids)].reset_index(drop=True)
        bx = bx[~bx.uid.isin(bad_uids)]
    man = man[(man.kind != "imagelevel") | man.label.isin(keep)].reset_index(drop=True)
    cls_idx = {c: i for i, c in enumerate(keep)}
    bx["cls"] = bx.cls_name.map(cls_idx)

    # ---- split
    animal = man[man.kind == "boxed"]
    if pc["split_by"] == "location":
        saved = paths.raw / "split_assignment.json"
        if saved.exists():  # fixed at download time (so the image-level test sample matches)
            assign = read_json(saved)
        else:
            loc_counts = pd.crosstab(animal.location, animal.label)
            assign = split_locations(loc_counts, pc["split_fractions"], pc["split_search_iters"], cfg["seed"])
        man["split"] = man.location.map(assign)
        # Sites with only empty images: spread them in proportion.
        rng = random.Random(cfg["seed"])
        for loc in man.loc[man.split.isna(), "location"].unique():
            r = rng.random()
            f = pc["split_fractions"]
            man.loc[man.location == loc, "split"] = "test" if r < f["test"] else ("val" if r < f["test"] + f["val"] else "train")
    else:
        rng = np.random.default_rng(cfg["seed"])
        f = pc["split_fractions"]
        man["split"] = rng.choice(list(SPLITS), size=len(man), p=[f[s] for s in SPLITS])

    # Image-level images are an independent test set at the test sites only.
    il = man.kind == "imagelevel"
    man.loc[il, "split"] = np.where(man.loc[il, "split"] == "test", "test_il", "unused")

    # Limit background images in train.
    tr_empty = man[(man.split == "train") & (man.label == "empty")]
    n_tr_animal = ((man.split == "train") & (man.label != "empty")).sum()
    max_bg = int(pc["train_empty_fraction"] / (1 - pc["train_empty_fraction"]) * n_tr_animal)
    if len(tr_empty) > max_bg:
        drop = tr_empty.sample(len(tr_empty) - max_bg, random_state=cfg["seed"]).index
        man.loc[drop, "split"] = "unused"

    summary = pd.crosstab(man.label, man.split)
    log.info("split summary (images):\n%s", summary.to_string())

    # ---- write tables
    paths.data.mkdir(parents=True, exist_ok=True)
    cols = ["uid", "lila_id", "rel", "path", "location", "datetime", "hour", "month", "season", "seq_id",
            "split", "label", "kind", "yolov4_seen", "n_boxes", "width", "height", "ir", "night", "green_frac", "vegetation",
            "contrast", "sharpness", "poor_visibility"]
    man[cols].to_csv(paths.manifest, index=False)
    bx.merge(man[["uid", "split"]], on="uid").to_csv(paths.boxes, index=False)
    tr = bx.merge(man[man.split == "train"][["uid"]], on="uid")
    write_json({
        "names": keep,
        "n_train_images": {c: int(tr[tr.cls_name == c].uid.nunique()) for c in keep},
        "n_train_boxes": {c: int((tr.cls_name == c).sum()) for c in keep},
        "dropped_rare_classes": dropped,
        "split_by": pc["split_by"],
        "split_locations": {s: sorted(man[man.split == s].location.unique().tolist()) for s in SPLITS},
    }, paths.classes)
    summary.to_csv(paths.data / "split_summary.csv")

    # ---- YOLO dataset
    by_uid = {u: g for u, g in bx.groupby("uid")}
    for s in SPLITS:
        sub = man[man.split == s]
        for r in sub.itertuples():
            ext = Path(r.path).suffix.lower() or ".jpg"
            _link(r.path, paths.yolo / "images" / s / f"{r.uid}{ext}", pc["link_mode"])
            lab = paths.yolo / "labels" / s / f"{r.uid}.txt"
            lab.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            if r.uid in by_uid:
                for b in by_uid[r.uid].itertuples():
                    cx = min(max((b.x + b.w / 2) / r.width, 0), 1)
                    cy = min(max((b.y + b.h / 2) / r.height, 0), 1)
                    lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {min(b.w / r.width, 1):.6f} {min(b.h / r.height, 1):.6f}")
            lab.write_text("\n".join(lines) + ("\n" if lines else ""))
    with open(paths.yolo / "data.yaml", "w") as f:
        yaml.safe_dump({"path": str(paths.yolo), "train": "images/train", "val": "images/val",
                        "test": "images/test", "names": {i: c for i, c in enumerate(keep)}}, f, sort_keys=False)

    # ---- COCO ground truth for the test split
    test = man[man.split == "test"].reset_index(drop=True)
    coco = {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": c} for i, c in enumerate(keep)]}
    for i, r in enumerate(test.itertuples(), 1):
        coco["images"].append({"id": i, "uid": r.uid, "width": int(r.width), "height": int(r.height)})
        for b in by_uid.get(r.uid, pd.DataFrame()).itertuples():
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": i,
                                        "category_id": int(b.cls) + 1, "bbox": [b.x, b.y, b.w, b.h],
                                        "area": b.w * b.h, "iscrowd": 0})
    write_json(coco, paths.coco_test)

    # ---- classifier crops (train/val) from ground-truth boxes
    jobs = []
    for s in ("train", "val"):
        sub = man[(man.split == s) & (man.label != "empty")]
        for r in sub.itertuples():
            b = by_uid[r.uid]
            jobs.append((r.path, r.uid, list(zip(b.cls, b.x, b.y, b.w, b.h)), str(paths.crops / s),
                         pc["crop_size"], pc["crop_margin"]))
    log.info("writing classifier crops for %d images ...", len(jobs))
    with Pool(pc["workers"]) as pool:
        n = sum(pool.map(_crop_job, jobs, chunksize=16))
    (paths.data / ".complete").touch()
    log.info("prepared: %d classes, %d crops, data.yaml at %s", len(keep), n, paths.yolo / "data.yaml")


if __name__ == "__main__":
    main()
