"""Build the simulation configuration from the real data and real models.

* Ecology from the full Nkhotakota metadata (all labelled images, not only
  the box subset): occupancy per species (MacKenzie model on weekly
  detection histories, allowing for unequal survey effort per site),
  independent-event rate per camera-day, night-activity share, empty
  (false) trigger rate per camera-day by season, and images per trigger.
* Detector error profiles fitted from each model's per-image test
  predictions (wildsim.calibrate), with measured inference time and energy
  and measured training energy.

Writes results/simulation/calibrated_config.json.

    python -m nkpipe.build_sim_config
"""

import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from wildsim import load_config as load_sim_config
from wildsim.calibrate import fit_profile
from wildsim.occupancy import mackenzie_counts

from . import ROOT, cli, load_classes, log, read_json, write_json
from .download import location_of, relative_name
from .prepare import norm_name

SIZE_KEYWORDS = {
    "large": ["elephant", "buffalo", "hippo", "eland", "sable", "roan", "kudu", "waterbuck", "zebra",
              "lion", "giraffe", "hartebeest", "wildebeest", "rhino", "nyala"],
    "small": ["genet", "civet", "mongoose", "hare", "squirrel", "rat", "mouse", "bird", "guineafowl",
              "francolin", "vervet", "monkey", "porcupine", "duiker", "grysbok", "galago", "dormouse",
              "rodent", "reptile", "lizard", "snake", "cane", "badger", "cat", "suni"],
}


def body_size(name):
    for size, kws in SIZE_KEYWORDS.items():
        if any(k in name for k in kws):
            return size
    return "medium"


def logit(p):
    p = min(max(p, 0.02), 0.98)
    return float(np.log(p / (1 - p)))


def ecology_from_metadata(paths, cfg, classes):
    """Species-level ecological parameters from all labelled Nkhotakota images."""
    meta = read_json(paths.metadata / "merged.json")
    dl, sc = cfg["download"], cfg["simulate"]
    cats = {c["id"]: norm_name(c["name"]) for c in meta["categories"]}
    labels = defaultdict(set)
    for a in meta["annotations"]:
        labels[a["image_id"]].add(cats[a["category_id"]])
    rows = []
    for im in meta["images"]:
        rel = relative_name(im["file_name"], dl).split("/")
        rows.append({"id": im["id"], "location": location_of(im),
                     "deployment": "/".join(rel[:2]) if len(rel) > 2 else rel[0],
                     "datetime": im.get("datetime"), "seq_id": im.get("seq_id"),
                     "labels": labels.get(im["id"], set())})
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df.datetime, errors="coerce")
    df = df[df.datetime.notna()]
    if df.empty:
        log.warning("metadata has no timestamps: ecology keeps default parameters")
        return None, {}
    df["date"] = df.datetime.dt.floor("D")
    df["wet"] = df.datetime.dt.month.isin([11, 12, 1, 2, 3, 4])
    night_start, night_end = cfg["prepare"]["night_hours"]
    df["night"] = (df.datetime.dt.hour >= night_start) | (df.datetime.dt.hour < night_end)

    # Survey effort: every day between the first and last image of each deployment.
    dep = df.groupby(["location", "deployment"]).date.agg(["min", "max"])
    days = {}
    for (loc, _), r in dep.iterrows():
        days.setdefault(loc, set()).update(pd.date_range(r["min"], r["max"], freq="D"))
    cam_days = {loc: len(d) for loc, d in days.items()}
    week = {loc: {(d - pd.Timestamp("2000-01-03")).days // sc["occasion_days"] for d in ds}
            for loc, ds in days.items()}
    locs = sorted(cam_days)
    J = np.array([len(week[l]) for l in locs])

    # Images per trigger (burst) and empty triggers per camera-day by season.
    burst = df.groupby("seq_id").size().median() if df.seq_id.notna().any() else 3
    burst = float(burst) if burst and burst > 0 else 3.0
    empty = df[df.labels.map(lambda s: s == {"empty"})]
    season_days = defaultdict(int)
    for loc, ds in days.items():
        for d in ds:
            season_days["wet" if d.month in (11, 12, 1, 2, 3, 4) else "dry"] += 1
    empty_rate = {s: len(empty[empty.wet == (s == "wet")]) / burst / max(season_days[s], 1) for s in ("dry", "wet")}

    gap = pd.Timedelta(minutes=sc["independence_minutes"])
    species = []
    for c in classes:
        sub = df[df.labels.map(lambda s: c in s)].sort_values("datetime")
        if sub.empty:
            species.append({"name": c, "_no_records": True})
            continue
        events = []
        for loc, g in sub.groupby("location"):
            new = g.datetime.diff().fillna(gap * 2) > gap
            events.append(g[new])
        ev = pd.concat(events)
        det_weeks = ev.assign(w=((ev.date - pd.Timestamp("2000-01-03")).dt.days // sc["occasion_days"]))
        d = np.array([det_weeks[det_weeks.location == l].w.nunique() for l in locs])
        fit = mackenzie_counts(d, J)
        occupied = ev.location.unique()
        rate = len(ev) / max(sum(cam_days.get(l, 0) for l in occupied), 1)
        species.append({"name": c, "psi_mackenzie": fit["psi"], "p_weekly": fit["p"],
                        "naive_occupancy": float((d > 0).mean()), "events": int(len(ev)),
                        "rate": float(rate), "night_prop": float(ev.night.mean())})
    survey = {"n_sites": len(locs), "median_camera_days": float(np.median(list(cam_days.values()))),
              "burst_size": burst, "empty_trigger_rate": empty_rate}
    return pd.DataFrame(species), survey


def main():
    args, cfg, paths = cli(__doc__)
    sc = cfg["simulate"]
    base = load_sim_config(ROOT / sc["base_config"])
    sim = copy.deepcopy(base)
    classes = load_classes(paths)
    n_train = read_json(paths.classes)["n_train_images"]
    out_dir = paths.results / "simulation"
    out_dir.mkdir(parents=True, exist_ok=True)

    eco, survey = ecology_from_metadata(paths, cfg, classes)
    defaults = {s["name"]: s for s in base["species"]}
    spp = []
    for c in classes:
        r = eco.set_index("name").loc[c].to_dict() if eco is not None and c in set(eco.name) else {}
        has = r and not r.get("_no_records") and not pd.isna(r.get("rate", np.nan))
        psi = r.get("psi_mackenzie") if has else None
        if psi is None or pd.isna(psi) or psi >= 0.999:  # degenerate fit: fall back to naive occupancy
            psi = r.get("naive_occupancy", 0.3) if has else 0.3
        spp.append({
            "name": c, "psi_logit": logit(psi), "b_veg": 0.0, "b_water": 0.0,
            "rate": float(r["rate"]) if has else defaults.get(c, {}).get("rate", 0.1),
            "night_prop": float(r["night_prop"]) if has else 0.5,
            "size": body_size(c), "n_train": max(int(n_train.get(c, 0)), 1),
            "priority": any(k in c for k in sc["priority_species_keywords"]),
        })
    sim["species"] = spp
    if survey:
        sim["survey"]["n_sites"] = int(survey["n_sites"])
        sim["survey"]["n_days"] = int(np.clip(survey["median_camera_days"], 30, 120))
        sim["survey"]["occasion_days"] = sc["occasion_days"]
        sim["survey"]["burst_size"] = max(1, int(round(survey["burst_size"])))
        for s in ("dry", "wet"):
            sim["seasons"][s]["empty_trigger_rate"] = float(survey["empty_trigger_rate"][s])
    if not any(s["priority"] for s in spp):
        log.warning("no priority species matched simulate.priority_species_keywords; marking the rarest")
        min(spp, key=lambda s: s["n_train"])["priority"] = True

    # ---- detector profiles from real predictions and measured energy
    def energy(name):
        p = paths.energy / name
        return read_json(p) if p.exists() else {}

    train_energy = {"YOLO11": energy("YOLO11_train.json"), "RT-DETR": energy("RT-DETR_train.json"),
                    "MegaDetector+classifier": energy("classifier_train.json"), "YOLOv4_reference": {}}
    models = {}
    for m in cfg["evaluate"]["models"]:
        pi_path = paths.eval / m / "per_image.csv"
        if not pi_path.exists():
            continue
        met = read_json(paths.eval / m / "metrics.json")
        t = met.get("timing", {})
        old = base["models"].get(m) or base["models"]["YOLO11"]
        infer = dict(old["infer_ms"])
        if t.get("ms_per_image"):
            infer["gpu"] = float(t["ms_per_image"])
        if t.get("cpu_ms_per_image"):
            infer["cpu"] = float(t["cpu_ms_per_image"])
        te = train_energy.get(m, {})
        prof = fit_profile(pd.read_csv(pi_path), spp, infer_ms=infer,
                           train_gpu_hours=te.get("gpu_hours") or old["train_gpu_hours"])
        prof["infer_kwh_per_image"] = t.get("kwh_per_image")
        prof["train_kwh"] = te.get("gpu_kwh")
        prof["_source"] = {"fitted_from": str(pi_path), "train_energy_measured": bool(te),
                           "edge_ms_assumed": True}
        models[m] = prof
        log.info("%s profile: recall_logit %.2f, night -%.2f, veg -%.2f, visibility -%.2f, fp/empty %.4f",
                 m, prof["recall_logit"], prof["night_penalty"], prof["veg_penalty"],
                 prof["rain_penalty"], prof["fp_per_empty"])
    if not models:
        raise SystemExit("no evaluated models found; run nkpipe.evaluate first")
    sim["models"] = models
    sim["footprint"]["train_kgco2_per_kwh"] = sc["energy_kgco2_per_kwh"]
    sim["footprint"]["grid_kgco2_per_kwh"]["gpu"] = sc["energy_kgco2_per_kwh"]
    sim["footprint"]["grid_kgco2_per_kwh"]["cpu"] = sc["energy_kgco2_per_kwh"]
    sim["_note"] = ("Generated by nkpipe.build_sim_config from Nkhotakota metadata and measured model "
                    "performance/energy. Monitoring-regime and vehicle parameters remain assumptions.")
    out = out_dir / "calibrated_config.json"
    Path(out).write_text(json.dumps(sim, indent=2, default=float) + "\n")
    if eco is not None:
        eco.to_csv(out_dir / "species_ecology_from_data.csv", index=False)
    write_json(survey, out_dir / "survey_from_data.json")
    log.info("wrote %s (%d species, %d models)", out, len(spp), len(models))


if __name__ == "__main__":
    main()
