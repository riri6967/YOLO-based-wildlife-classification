"""True ecological state and environmental conditions of a simulated reserve.

The world is generated once per replicate and shared by every detector, so
model comparisons are paired (common random numbers): differences between
models come from detection alone, not from different simulated animals.
"""

from dataclasses import dataclass

import numpy as np

from . import expit


@dataclass
class World:
    season: str
    veg: np.ndarray            # (S,) site vegetation density, 0-1
    water_dist: np.ndarray     # (S,) km to nearest water
    veg_eff: np.ndarray        # (S, D) effective vegetation incl. seasonal growth
    rain: np.ndarray           # (S, D) 0/1 rain on site-day
    psi: np.ndarray            # (S, K) true occupancy probability
    z: np.ndarray              # (S, K) true occupancy state
    n_day: np.ndarray          # (S, D, K) daytime animal passages
    n_night: np.ndarray        # (S, D, K) night-time animal passages
    trig_day: np.ndarray       # (S, D, K) passages that triggered the camera (day)
    trig_night: np.ndarray     # (S, D, K) passages that triggered the camera (night)
    empty: np.ndarray          # (S, D) empty (false) camera triggers

    @property
    def n_passages(self):
        return self.n_day + self.n_night


def simulate_world(cfg, season, rng):
    sv, env, spp = cfg["survey"], cfg["environment"], cfg["species"]
    sea = cfg["seasons"][season]
    S, D, K = sv["n_sites"], sv["n_days"], len(spp)

    veg = rng.beta(*env["veg_beta"], size=S)
    water = rng.gamma(*env["water_dist_gamma"], size=S)
    veg_c = (veg - veg.mean()) / (veg.std() + 1e-9)
    water_c = (water - water.mean()) / (water.std() + 1e-9)

    # Rain is mostly a regional (day-level) process with some local variation.
    rain_day = rng.random(D) < sea["rain_prob"]
    local = rng.random((S, D)) < 0.8
    rain = (rain_day[None, :] & local).astype(int)
    veg_eff = np.clip(veg[:, None] + sea["veg_boost"], 0, 1) * np.ones((1, D))

    psi0 = np.array([s["psi_logit"] for s in spp])
    bv = np.array([s["b_veg"] for s in spp])
    bw = np.array([s["b_water"] for s in spp])
    psi = expit(psi0[None, :] + veg_c[:, None] * bv + water_c[:, None] * bw)
    z = (rng.random((S, K)) < psi).astype(int)

    rate = np.array([s["rate"] for s in spp]) * sea["activity_mult"]
    night_prop = np.array([s["night_prop"] for s in spp])
    lam = z[:, None, :] * rate[None, None, :] * np.ones((S, D, K))
    n = rng.poisson(lam)
    n_night = rng.binomial(n, night_prop[None, None, :])
    n_day = n - n_night

    # Camera hardware (PIR trigger) misses are model independent.
    p_trig = np.array([env["trigger_prob"][s["size"]] for s in spp])
    p_trig_night = np.clip(p_trig - env["trigger_night_penalty"], 0, 1)
    trig_day = rng.binomial(n_day, p_trig[None, None, :])
    trig_night = rng.binomial(n_night, p_trig_night[None, None, :])

    # Moving grass and rain cause empty triggers, more so in dense, wet habitat.
    empty_lam = sea["empty_trigger_rate"] * (1 + env["empty_veg_mult"] * veg_eff
                                             + env["empty_rain_mult"] * rain)
    empty = rng.poisson(empty_lam)

    return World(season, veg, water, veg_eff, rain, psi, z,
                 n_day, n_night, trig_day, trig_night, empty)
