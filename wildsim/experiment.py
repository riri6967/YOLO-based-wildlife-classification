"""Factorial experiment: detector x confidence threshold x season x
monitoring regime, replicated over simulated worlds."""

import numpy as np
import pandas as pd

from . import occupancy as occ
from .detection import observe
from .footprint import survey_footprint
from .monitoring import alert_outcomes
from .world import simulate_world


def condition_recall(world, obs):
    """Realised recall of animal passages, split by environmental condition."""
    n = world.n_passages.sum(axis=2)
    c = obs.correct.sum(axis=2)
    dense = world.veg_eff > 0.6
    rain = world.rain == 1
    n_night = world.n_night.sum()
    n_day = world.n_day.sum()
    c_night = obs.correct_night.sum()

    def r(mask):
        return c[mask].sum() / n[mask].sum() if n[mask].sum() else np.nan

    return {
        "recall_all": c.sum() / max(n.sum(), 1),
        "recall_day": (c.sum() - c_night) / max(n_day, 1),
        "recall_night": c_night / max(n_night, 1),
        "recall_dense_veg": r(dense),
        "recall_open_veg": r(~dense),
        "recall_rain": r(rain),
        "recall_dry_day": r(~rain),
    }


def evaluate(cfg, world, obs, regimes, rng, true_richness):
    spp = cfg["species"]
    L = cfg["survey"]["occasion_days"]
    y = obs.y
    species_rows = []
    for k, s in enumerate(spp):
        yk = occ.to_occasions(y[..., k], L)
        true_psi = world.z[:, k].mean()
        mk = occ.mackenzie(yk)
        rl = occ.royle_link(yk)
        species_rows.append({
            "species": s["name"], "n_train": s["n_train"], "priority": s["priority"],
            "true_occupancy": true_psi,
            "naive": occ.naive(yk), "mackenzie": mk["psi"], "royle_link": rl["psi"],
            "est_p": mk["p"], "est_p10": rl["p10"],
        })
    sp = pd.DataFrame(species_rows)
    for m in ("naive", "mackenzie", "royle_link"):
        sp[f"bias_{m}"] = sp[m] - sp["true_occupancy"]

    obs_richness = int((y.sum(axis=(0, 1)) > 0).sum())
    site_rich_true = world.z.sum(axis=1)
    site_rich_obs = (y.sum(axis=1) > 0).sum(axis=1)

    base = {
        "richness_bias": obs_richness - true_richness,
        "site_richness_bias": float((site_rich_obs - site_rich_true).mean()),
        "mae_occ_naive": sp["bias_naive"].abs().mean(),
        "mae_occ_mackenzie": sp["bias_mackenzie"].abs().mean(),
        "mae_occ_royle_link": sp["bias_royle_link"].abs().mean(),
        "false_record_share": obs.false_pos.sum() / max(obs.false_pos.sum() + obs.correct.sum(), 1),
        "n_images": obs.n_images,
        **condition_recall(world, obs),
    }
    rows = []
    for reg in regimes:
        al = alert_outcomes(cfg, world, obs, reg, rng)
        fpr = survey_footprint(cfg, obs.model, obs.n_images, al, reg)
        fpr["co2_per_detected_event_kg"] = fpr["co2_total_kg"] / max(obs.correct.sum(), 1)
        rows.append({"regime": reg, **base, **al, **fpr})
    return sp, rows


def run(cfg, models=None, thresholds=(0.25, 0.5, 0.75), seasons=("dry", "wet"),
        regimes=None, n_reps=10, seed=1):
    models = models or list(cfg["models"])
    regimes = regimes or list(cfg["monitoring_regimes"])
    ss = np.random.SeedSequence(seed)
    scen, spec = [], []
    for season in seasons:
        for rep, child in enumerate(ss.spawn(n_reps)):
            world_rng, det_seed = np.random.default_rng(child), child.spawn(1)[0]
            world = simulate_world(cfg, season, world_rng)
            true_richness = int((world.z.sum(axis=0) > 0).sum())
            for mi, model in enumerate(models):
                for ti, thr in enumerate(thresholds):
                    rng = np.random.default_rng([det_seed.entropy, mi, ti, rep])
                    obs = observe(cfg, world, model, thr, rng)
                    sp, rows = evaluate(cfg, world, obs, regimes, rng, true_richness)
                    key = {"season": season, "rep": rep, "model": model, "threshold": thr}
                    spec.append(sp.assign(**key))
                    scen.extend({**key, **r} for r in rows)
    return pd.DataFrame(scen), pd.concat(spec, ignore_index=True)
