"""Download Nkhotakota metadata and the images needed for the study.

1. Find the COCO Camera Traps metadata via LILA's dataset index (or use the
   URLs given in the config), download and merge it.
2. Select every box-annotated image plus a location-stratified sample of
   empty images, and download them from the public Google Cloud bucket.
3. Optionally fetch the NWR_YOLO_v1 (YOLOv4) reference model from Zenodo.

    python -m nkpipe.download --config configs/pipeline.yaml [--yolov4] [--metadata-only]
"""

import csv
import io
import json
import random
import shutil
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import cli, log, read_json, write_json

UA = {"User-Agent": "nkpipe/1.0 (camera-trap research)"}


def fetch(url, dest=None, retries=3, timeout=60, progress=False):
    """Download `url` (to memory, or to `dest`). File downloads resume from a
    partial `.part` file after a dropped connection instead of starting over."""
    import time
    last = None
    for attempt in range(retries):
        try:
            headers = dict(UA)
            tmp = Path(str(dest) + ".part") if dest is not None else None
            have = tmp.stat().st_size if tmp is not None and tmp.exists() else 0
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if dest is None:
                    return r.read()
                tmp.parent.mkdir(parents=True, exist_ok=True)
                if have and r.status != 206:  # server ignored Range: start over
                    have = 0
                total = int(r.headers.get("Content-Length") or 0) + have
                done, next_log = have, have + (50 << 20)
                with open(tmp, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if progress and done >= next_log:
                            log.info("  %s: %.0f / %.0f MB", Path(dest).name, done / 1e6, total / 1e6)
                            next_log += 50 << 20
                if total and done < total:
                    raise IOError(f"incomplete download ({done} of {total} bytes)")
                tmp.rename(dest)
                return dest
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if e.code == 416 and tmp is not None and tmp.exists():
                tmp.rename(dest)  # Range starts past the end: the .part file is already complete
                return dest
            last = e
        except Exception as e:  # network hiccup: retry (and resume)
            last = e
        if attempt == retries - 1:
            break
        if progress:
            log.warning("  download interrupted (%s); resuming (attempt %d/%d)", last, attempt + 2, retries)
        time.sleep(min(2 ** attempt, 30))
    raise last


def discover_metadata_urls(dl):
    if dl.get("metadata_urls"):
        return list(dl["metadata_urls"])
    rows = None
    for url in dl.get("index_urls") or [dl["index_url"]]:
        try:
            log.info("reading LILA index %s", url)
            rows = list(csv.DictReader(io.StringIO(fetch(url).decode("utf-8", "replace"))))
            break
        except Exception as e:
            log.warning("could not read %s (%s)", url, e)
    if rows is None:
        raise SystemExit("Could not read the LILA index. Open https://lila.science/datasets/nkhotakota-camera-traps/, "
                         "copy the metadata link(s) and put them under download.metadata_urls in the config.")
    match = [r for r in rows if any(dl["dataset_match"] in str(v).lower() for v in r.values())]
    if not match:
        raise SystemExit("Dataset not found in LILA index; set download.metadata_urls in the config.")
    urls = []
    for k, v in match[0].items():
        v = (v or "").strip()
        if not v or "image_base" in k.lower() or "md_results" in k.lower() or "preview" in k.lower():
            continue
        if v.lower().endswith((".json", ".zip")) and ("metadata" in k.lower() or "bbox" in k.lower()):
            if v.startswith("http"):
                urls.append(v)
            else:
                urls += [b.rstrip("/") + "/" + v.lstrip("/") for b in dl["relative_url_bases"]]
    if not urls:
        raise SystemExit(f"No metadata URL in LILA index row {match[0]}; set download.metadata_urls.")
    return urls


def load_coco_from_bytes(name, data):
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            jsons = [n for n in z.namelist() if n.lower().endswith(".json")]
            return [json.loads(z.read(n)) for n in jsons]
    return [json.loads(data)]


def merge_coco(dicts):
    images, cats, anns, seen = {}, {}, [], set()
    for d in dicts:
        for im in d.get("images", []):
            images.setdefault(im["id"], {}).update(im)
        for c in d.get("categories", []):
            cats[c["id"]] = c
        for a in d.get("annotations", []):
            key = (a["image_id"], a["category_id"], tuple(a.get("bbox") or ()))
            if key not in seen:
                seen.add(key)
                anns.append(a)
    return {"images": list(images.values()), "categories": list(cats.values()), "annotations": anns}


def get_metadata(dl, paths):
    out = paths.metadata / "merged.json"
    if out.exists():
        return read_json(out)
    dicts, got = [], set()
    for url in discover_metadata_urls(dl):
        name = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
        if name in got:
            continue
        try:
            data = fetch(url)
        except Exception as e:
            log.warning("could not fetch %s (%s)", url, e)
            continue
        (paths.metadata / name).write_bytes(data)
        dicts += load_coco_from_bytes(name, data)
        got.add(name)
        log.info("metadata: %s", name)
    if not dicts:
        raise SystemExit("No metadata downloaded; set download.metadata_urls in the config.")
    merged = merge_coco(dicts)
    write_json(merged, out)
    return merged


def location_of(im):
    return str(im.get("location") or im["file_name"].replace("\\", "/").split("/")[0])


def yolov4_seen_names(paths):
    """Basenames of images the YOLOv4 reference was trained or validated on (Zenodo lists)."""
    d = paths.models / "yolov4_reference"
    seen = set()
    for name in ("train.txt", "validation.txt"):
        p = d / name
        if p.exists():
            seen |= {l.strip().replace("\\", "/").rsplit("/", 1)[-1] for l in open(p) if l.strip()}
    return seen


def select_images(meta, cfg, paths):
    """Box-annotated images, a location-stratified sample of empty images, and an
    independent image-level test sample (test sites only, never seen by YOLOv4).

    The camera-site split is fixed here from the metadata and saved, so the
    image-level test sample can be drawn from the test sites only."""
    from .prepare import norm_name, split_locations  # local import: prepare imports this module

    dl, pc = cfg["download"], cfg["prepare"]
    cats = {c["id"]: norm_name(c["name"]) for c in meta["categories"]}
    exclude = {norm_name(c) for c in pc["exclude_categories"]}
    by_img = defaultdict(list)
    for a in meta["annotations"]:
        by_img[a["image_id"]].append(a)
    boxed, empty, imagelevel = [], defaultdict(list), []
    for im in meta["images"]:
        anns = by_img.get(im["id"], [])
        names = {cats[a["category_id"]] for a in anns}
        if any(a.get("bbox") for a in anns):
            boxed.append(im)
        elif names and names <= {"empty"}:
            empty[location_of(im)].append(im)
        elif len(names) == 1 and not names & exclude:
            imagelevel.append(im)
    rng = random.Random(cfg["seed"])
    if dl.get("max_boxed_images"):
        rng.shuffle(boxed)
        boxed = boxed[: dl["max_boxed_images"]]

    # Camera-site split from the box-annotated images (dominant species per image).
    dom = []
    for im in boxed:
        c = Counter(cats[a["category_id"]] for a in by_img[im["id"]] if a.get("bbox")).most_common(1)[0][0]
        dom.append((location_of(im), c))
    import pandas as pd
    loc_counts = pd.crosstab(pd.Series([d[0] for d in dom], name="location"),
                             pd.Series([d[1] for d in dom], name="label"))
    assign = split_locations(loc_counts, pc["split_fractions"], pc["split_search_iters"], cfg["seed"])
    write_json(assign, paths.raw / "split_assignment.json")

    n_total = sum(len(v) for v in empty.values())
    want = min(dl["n_empty_images"], n_total)
    chosen = []
    for loc, ims in empty.items():
        k = round(want * len(ims) / max(n_total, 1))
        chosen += rng.sample(ims, min(k, len(ims)))

    seen = yolov4_seen_names(paths)
    pool = [im for im in imagelevel if assign.get(location_of(im)) == "test"
            and relative_name(im["file_name"], dl).rsplit("/", 1)[-1] not in seen]
    il = rng.sample(pool, min(dl.get("n_imagelevel_test_images", 0), len(pool)))
    log.info("selected %d box-annotated, %d empty and %d independent image-level test images "
             "(YOLOv4 training list: %d images)", len(boxed), len(chosen), len(il), len(seen))
    return boxed + chosen + il


def relative_name(file_name, dl):
    """file_name relative to the image bucket folder (LILA names may include the folder)."""
    prefix = dl["image_base_url"].rstrip("/").rsplit("/", 1)[-1] + "/"
    fn = file_name.replace("\\", "/")
    return fn[len(prefix):] if fn.startswith(prefix) else fn


def download_images(images, dl, paths):
    base = dl["image_base_url"].rstrip("/") + "/"

    def one(im):
        rel = relative_name(im["file_name"], dl)
        dest = paths.images / rel
        if dest.exists() and dest.stat().st_size > 0:
            return im["id"], "exists"
        try:
            fetch(base + urllib.parse.quote(rel), dest)
            return im["id"], "ok"
        except urllib.error.HTTPError as e:
            return im["id"], f"http_{e.code}"
        except Exception as e:
            return im["id"], f"error_{type(e).__name__}"

    status = {}
    with ThreadPoolExecutor(dl["workers"]) as ex:
        futs = [ex.submit(one, im) for im in images]
        for i, f in enumerate(as_completed(futs), 1):
            k, s = f.result()
            status[k] = s
            if i % 2000 == 0 or i == len(futs):
                log.info("images: %d/%d", i, len(futs))
    with open(paths.raw / "download_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "status"])
        w.writerows(status.items())
    bad = sum(1 for s in status.values() if s not in ("ok", "exists"))
    log.info("download finished: %d failed (humans were removed from LILA, so some 404s are expected)", bad)


def download_yolov4(record, paths):
    dest = paths.models / "yolov4_reference"
    dest.mkdir(parents=True, exist_ok=True)
    info = json.loads(fetch(f"https://zenodo.org/api/records/{record}"))
    for f in info["files"]:
        name = f["key"]
        out = dest / name
        if not out.exists():
            log.info("zenodo: %s (%.1f MB)", name, f.get("size", 0) / 1e6)
            fetch(f["links"]["self"], out, retries=20, progress=True)
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(out) as z:
                z.extractall(dest)
    log.info("YOLOv4 reference files in %s", dest)


def main():
    args, cfg, paths = cli(__doc__, lambda ap: (
        ap.add_argument("--yolov4", action="store_true", help="also fetch the Zenodo YOLOv4 model"),
        ap.add_argument("--metadata-only", action="store_true")))
    dl = cfg["download"]
    meta = get_metadata(dl, paths)
    log.info("metadata: %d images, %d annotations, %d categories",
             len(meta["images"]), len(meta["annotations"]), len(meta["categories"]))
    if args.yolov4:
        download_yolov4(dl["yolov4_zenodo_record"], paths)
    if args.metadata_only:
        return
    sel = select_images(meta, cfg, paths)
    write_json([im["id"] for im in sel], paths.raw / "selected_ids.json")
    log.info("approx. download size: %.1f GB", len(sel) * 0.7 / 1000)
    download_images(sel, dl, paths)


if __name__ == "__main__":
    main()
