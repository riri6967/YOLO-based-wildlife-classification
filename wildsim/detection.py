"""Apply a detector's error profile to a simulated world.

A model profile describes, on the logit scale, how the probability of a
correct species-level detection depends on training-set size, body size,
confidence threshold and the environmental conditions of each image
(night/IR, vegetation density, rain). False positives arise from empty
triggers, whose number is itself driven by vegetation and rain, so the
environment affects both missed animals and spurious records.
"""

from dataclasses import dataclass

import numpy as np

from . import expit


@dataclass
class Observation:
    model: str
    threshold: float
    correct: np.ndarray     # (S, D, K) passages detected and labelled correctly
    correct_night: np.ndarray  # (S, D, K) subset of `correct` recorded at night (IR)
    false_pos: np.ndarray   # (S, D, K) spurious records (empty triggers + misclassifications)
    n_images: int

    @property
    def y(self):
        """Daily detection/non-detection record as a monitoring programme sees it."""
        return ((self.correct + self.false_pos) > 0).astype(int)


def species_terms(cfg, prof, threshold):
    spp = cfg["species"]
    n_train = np.array([s["n_train"] for s in spp], dtype=float)
    data = np.log10(n_train / 1000.0)
    size = np.array([prof["size_effect"][s["size"]] for s in spp])
    base = (prof["recall_logit"] + prof["data_slope"] * data + size
            - prof["thr_recall_slope"] * (threshold - 0.5))
    mis = np.clip(prof["misclass"] + prof["misclass_data_slope"] * data, 0.01, 0.5)
    label_w = n_train / n_train.sum()  # models over-predict well-represented classes
    return base, mis, label_w


def detection_prob(prof, base, night, veg_eff, rain):
    """P(correct detection | camera triggered) for every (site, day, species)."""
    lg = (base[None, None, :]
          - prof["night_penalty"] * night
          - prof["veg_penalty"] * veg_eff[:, :, None]
          - prof["rain_penalty"] * rain[:, :, None])
    return expit(lg)


def observe(cfg, world, model_name, threshold, rng):
    prof = cfg["models"][model_name]
    base, mis, label_w = species_terms(cfg, prof, threshold)
    K = len(base)

    p_day = detection_prob(prof, base, 0, world.veg_eff, world.rain)
    p_night = detection_prob(prof, base, 1, world.veg_eff, world.rain)
    det_day = rng.binomial(world.trig_day, p_day)
    det_night = rng.binomial(world.trig_night, p_night)

    wrong_day = rng.binomial(det_day, mis[None, None, :])
    wrong_night = rng.binomial(det_night, mis[None, None, :])
    wrong = wrong_day + wrong_night
    correct_night = det_night - wrong_night
    correct = det_day - wrong_day + correct_night

    fp = np.zeros_like(correct)
    # Misclassified animals become records of some other species.
    for k in range(K):
        if wrong[..., k].any():
            w = label_w.copy()
            w[k] = 0
            fp += rng.multinomial(wrong[..., k], w / w.sum())

    p_fp = (prof["fp_per_empty"] * np.exp(-prof["thr_fp_slope"] * (threshold - 0.5))
            * (1 + (prof["fp_veg_mult"] - 1) * world.veg_eff))
    n_fp = rng.binomial(world.empty, np.clip(p_fp, 0, 1))
    fp += rng.multinomial(n_fp, label_w)

    burst = cfg["survey"]["burst_size"]
    n_images = int(burst * (world.trig_day.sum() + world.trig_night.sum() + world.empty.sum()))
    return Observation(model_name, threshold, correct, correct_night, fp, n_images)
