"""Global (Sobol) sensitivity analysis of monitoring outcomes.

Detector quality, environmental conditions and monitoring design are varied
together, so the indices show which factor contributes most to biased
occupancy estimates, missed priority events, slow responses and emissions.
"""

import copy

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample

from .experiment import evaluate
from .detection import observe
from .world import simulate_world

PROBLEM = {
    "names": [
        "recall_logit",        # overall detector quality
        "night_penalty",       # IR / night-time degradation
        "veg_penalty",         # occlusion by vegetation
        "fp_per_empty",        # false-positive rate on empty triggers
        "threshold",           # confidence threshold
        "rain_prob",           # weather
        "empty_trigger_rate",  # moving-grass false triggers (habitat/season)
        "check_interval_days", # SD-card servicing interval (0 = real-time)
        "response_hours",      # field response delay
        "n_sites",             # sampling effort
    ],
    "bounds": [
        [0.5, 3.0], [0.0, 1.5], [0.0, 1.5], [0.0, 0.05], [0.2, 0.8],
        [0.0, 0.5], [1.0, 15.0], [0.0, 28.0], [1.0, 48.0], [20, 120],
    ],
}
PROBLEM["num_vars"] = len(PROBLEM["names"])

OUTPUTS = ["mae_occ_mackenzie", "mae_occ_royle_link", "missed_event_frac",
           "median_latency_h", "false_alarms", "co2_total_kg"]


def _one_run(cfg, x, base_model, rng):
    p = dict(zip(PROBLEM["names"], x))
    c = copy.deepcopy(cfg)
    m = c["models"][base_model]
    for k in ("recall_logit", "night_penalty", "veg_penalty", "fp_per_empty"):
        m[k] = float(p[k])
    c["seasons"]["sim"] = {**c["seasons"]["wet"], "rain_prob": float(p["rain_prob"]),
                           "empty_trigger_rate": float(p["empty_trigger_rate"])}
    c["survey"]["n_sites"] = int(round(p["n_sites"]))
    check = float(p["check_interval_days"])
    c["monitoring_regimes"] = {"sim": {
        "check_interval_days": 0 if check < 1 else int(round(check)),
        "transmit_hours": 0.5 if check < 1 else 0.0,
        "processing_hours": 0.2 if check < 1 else 24.0,
        "response_hours": float(p["response_hours"]),
        "service_interval_days": 30,
    }}
    world = simulate_world(c, "sim", rng)
    obs = observe(c, world, base_model, float(p["threshold"]), rng)
    true_rich = int((world.z.sum(axis=0) > 0).sum())
    _, rows = evaluate(c, world, obs, ["sim"], rng, true_rich)
    return rows[0]


def run_sobol(cfg, n_base=64, base_model="YOLO11", seed=7):
    X = sobol_sample.sample(PROBLEM, n_base, calc_second_order=False, seed=seed)
    rng = np.random.default_rng(seed)
    Y = pd.DataFrame([_one_run(cfg, x, base_model, rng) for x in X])
    out = []
    for o in OUTPUTS:
        y = Y[o].astype(float).fillna(Y[o].mean()).to_numpy()
        si = sobol.analyze(PROBLEM, y, calc_second_order=False, seed=seed)
        for name, s1, st, st_ci in zip(PROBLEM["names"], si["S1"], si["ST"], si["ST_conf"]):
            out.append({"output": o, "parameter": name, "S1": s1, "ST": st, "ST_conf": st_ci})
    return pd.DataFrame(out), pd.concat([pd.DataFrame(X, columns=PROBLEM["names"]), Y], axis=1)
