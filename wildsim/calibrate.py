"""Fit a detector error profile from real test-set predictions.

This is the bridge between the computer-vision evaluation and the
simulation: train each model on Nkhotakota, run it on the held-out
(camera-site-disjoint) test set, export one row per image, and fit the
profile that `detection.observe` uses.

Expected CSV columns
  true_label   species name or "empty"
  pred_label   top predicted species or "empty"
  score        confidence of the top prediction (0-1)
  night        1 if the image is an IR/night image, else 0
  vegetation   optional, 0-1 vegetation density in the frame / at the site
  rain         optional, 1 if rain/fog/lens water is visible
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit

REF_THRESHOLD = 0.5


def _logistic(X, y, l2=1e-3, fixed_offset=None):
    off = np.zeros(len(y)) if fixed_offset is None else fixed_offset

    def nll(b):
        z = X @ b + off
        return -(y * log_expit(z) + (1 - y) * log_expit(-z)).sum() + l2 * (b ** 2).sum()

    return minimize(nll, np.zeros(X.shape[1]), method="L-BFGS-B").x


def fit_profile(df, species_cfg, infer_ms=None, train_gpu_hours=None):
    df = df.copy()
    for col in ("vegetation", "rain"):
        if col not in df:
            df[col] = 0.0
    meta = {s["name"]: s for s in species_cfg}
    ani = df[df.true_label != "empty"].copy()
    ani = ani[ani.true_label.isin(meta)]
    ani["data"] = np.log10(ani.true_label.map(lambda s: meta[s]["n_train"]) / 1000.0)
    size = ani.true_label.map(lambda s: meta[s]["size"])
    X = np.column_stack([np.ones(len(ani)), ani.data, (size == "small"), (size == "large"),
                         ani.night, ani.vegetation, ani.rain]).astype(float)

    def success(t):
        return ((ani.pred_label == ani.true_label) & (ani.score >= t)).to_numpy(float)

    b = _logistic(X, success(REF_THRESHOLD))
    # Threshold slope: refit only the intercept at other thresholds.
    rest = X[:, 1:] @ b[1:]
    ints = {t: _logistic(X[:, :1], success(t), fixed_offset=rest)[0] for t in (0.25, 0.75)}
    thr_recall_slope = -(ints[0.75] - ints[0.25]) / 0.5

    called = ani[(ani.pred_label != "empty") & (ani.score >= REF_THRESHOLD)]
    wrong = (called.pred_label != called.true_label).to_numpy(float)
    A = np.column_stack([np.ones(len(called)), called.data])
    mis_int, mis_slope = np.linalg.lstsq(A, wrong, rcond=None)[0]

    emp = df[df.true_label == "empty"]

    def fp_rate(t):
        return max(((emp.pred_label != "empty") & (emp.score >= t)).mean(), 1e-6)

    # P(false call) = fp0 * (1 + (m - 1) * vegetation): linear in vegetation.
    hit = ((emp.pred_label != "empty") & (emp.score >= REF_THRESHOLD)).to_numpy(float)
    if emp.vegetation.nunique() > 1:
        a, b_v = np.linalg.lstsq(np.column_stack([np.ones(len(emp)), emp.vegetation]),
                                 hit, rcond=None)[0]
        fp0, fp_veg_mult = max(a, 1e-6), max(1 + b_v / max(a, 1e-6), 0.0)
    else:
        fp0, fp_veg_mult = fp_rate(REF_THRESHOLD), 1.0

    return {
        "recall_logit": float(b[0]),
        "data_slope": float(b[1]),
        "size_effect": {"small": float(b[2]), "medium": 0.0, "large": float(b[3])},
        "night_penalty": float(-b[4]),
        "veg_penalty": float(-b[5]),
        "rain_penalty": float(-b[6]),
        "fp_per_empty": float(fp0),
        "fp_veg_mult": float(fp_veg_mult),
        "misclass": float(np.clip(mis_int, 0.0, 0.5)),
        "misclass_data_slope": float(mis_slope),
        "thr_recall_slope": float(thr_recall_slope),
        "thr_fp_slope": float(np.log(fp_rate(0.25) / fp_rate(0.75)) / 0.5),
        "infer_ms": infer_ms or {"gpu": np.nan, "cpu": np.nan, "edge": np.nan},
        "train_gpu_hours": train_gpu_hours,
    }


def synthetic_predictions(prof, species_cfg, n=20000, empty_frac=0.4, seed=0):
    """Generate fake prediction rows from a known profile (for testing calibration)."""
    rng = np.random.default_rng(seed)
    names = [s["name"] for s in species_cfg]
    meta = {s["name"]: s for s in species_cfg}
    rows = []
    for _ in range(n):
        night, veg, rain = rng.random() < 0.4, rng.random(), rng.random() < 0.2
        if rng.random() < empty_frac:
            fp = prof["fp_per_empty"] * (1 + (prof["fp_veg_mult"] - 1) * veg)
            # Invert P(score >= t) = fp * exp(-k (t - 0.5)) to draw a score.
            score = 0.5 + np.log(fp / rng.random()) / prof["thr_fp_slope"]
            called = score > 0
            rows.append(("empty", rng.choice(names) if called else "empty",
                         float(np.clip(score, 0, 1)), night, veg, rain))
            continue
        sp = rng.choice(names)
        s = meta[sp]
        lg = (prof["recall_logit"] + prof["data_slope"] * np.log10(s["n_train"] / 1000)
              + prof["size_effect"][s["size"]] - prof["night_penalty"] * night
              - prof["veg_penalty"] * veg - prof["rain_penalty"] * rain)
        # Score at which this animal stops being a correct call: logistic in threshold.
        t_star = 0.5 + (lg - np.log(1 / rng.random() - 1)) / prof["thr_recall_slope"]
        rows.append((sp, sp, float(np.clip(t_star, 0, 1)), night, veg, rain))
    return pd.DataFrame(rows, columns=["true_label", "pred_label", "score", "night", "vegetation", "rain"])
